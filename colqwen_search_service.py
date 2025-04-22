from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

import modal
from pydantic import BaseModel

# Modal setup
app = modal.App("colqwen-search-service")
MINUTES = 60  # seconds
CACHE_DIR = "/hf-cache"

# Images setup
model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        [
            "pydantic==2.5.2",
            "git+https://github.com/illuin-tech/colpali.git@782edcd50108d1842d154730ad3ce72476a2d17d",
            "hf_transfer==0.1.8",
            "qwen-vl-utils==0.0.8",
            "torchvision==0.19.1",
            "fastapi==0.115.4",
            "python-multipart==0.0.6",
            "uvicorn==0.27.1",
            "supabase==2.9.0",
            "pillow==10.4.0",
            "numpy==1.24.3",
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": CACHE_DIR})
)

pdf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("poppler-utils")
    .pip_install("pdf2image==1.17.0", "pillow==10.4.0", "pydantic==2.5.2")
)

# Make sure all functions use the same base image with required dependencies
base_image = model_image

# Volumes setup
pdf_volume = modal.Volume.from_name("colqwen-search-pdfs", create_if_missing=True)
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
PDF_ROOT = Path("/vol/pdfs/")

# Model constants
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "aca78372505e6cb469c4fa6a35c60265b00ff5a4"

# Vector dimensions constants
EMBEDDING_DIMENSION = 1536  # Define the embedding dimension as a constant

# Supabase setup
from modal.secret import Secret

supabase_secret = Secret.from_name("supabase-credentials")


# Data models for API responses
class EmbeddingResponse(BaseModel):
    file_id: UUID
    status: str
    message: str


class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5


class SearchResult(BaseModel):
    file_id: UUID
    page_number: int
    score: float
    content_preview: Optional[str] = None
    text_content: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]


class QuestionResponse(BaseModel):
    answer: str
    sources: Optional[SearchResponse] = None


class FileInfo(BaseModel):
    file_id: UUID
    filename: str
    size: int
    created_at: str
    status: str


class FileResponse(BaseModel):
    files: List[FileInfo]


class FileDetailResponse(FileInfo):
    embedding_count: int
    last_updated: str


@app.function(
    image=model_image, volumes={CACHE_DIR: cache_volume}, timeout=20 * MINUTES
)
def download_model():
    with model_image.imports():
        from huggingface_hub import snapshot_download

    result = snapshot_download(
        MODEL_NAME,
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    print(f"Downloaded model weights to {result}")


@app.function(image=pdf_image)
def convert_pdf_to_images(pdf_bytes):
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf_bytes, fmt="jpeg")
    return images


@app.cls(
    image=model_image,
    gpu="A100-80GB",
    scaledown_window=10 * MINUTES,  # spin down when inactive
    volumes={"/vol/pdfs/": pdf_volume, CACHE_DIR: cache_volume},
    secrets=[supabase_secret],
)
class ColQwenModel:
    @modal.enter()
    def load_models(self):
        with model_image.imports():
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        # Initialize models
        self.colqwen2_model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.colqwen2_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
        self.qwen2_vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
        )
        self.qwen2_vl_model.to("cuda:0")
        self.qwen2_vl_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
        )

        # Initialize Supabase client
        from supabase import create_client
        import os

        self.supabase = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_KEY")
        )

    def _process_embedding_tensor(self, embedding_tensor):
        """
        Standardized function to convert ColQwen2 embedding tensors to vectors suitable for database storage
        """
        with model_image.imports():
            import torch
            import numpy as np

        # Process tensor to get consistent format for pgvector
        vector = embedding_tensor.detach().cpu().float().flatten().numpy()[:EMBEDDING_DIMENSION].tolist()
        return vector

    @modal.method()
    def generate_embeddings_for_file(self, file_id: str, pdf_bytes: bytes):
        """Generate embeddings for a single PDF file"""
        with model_image.imports():
            import torch
            import numpy as np

        # Convert PDF to images
        images = modal.Function.lookup("colqwen-search-service", "convert_pdf_to_images").remote(pdf_bytes)

        # Store images on Volume for later retrieval
        session_dir = PDF_ROOT / file_id
        session_dir.mkdir(exist_ok=True, parents=True)
        for i, image in enumerate(images):
            filename = session_dir / f"{str(i).zfill(3)}.jpg"
            image.save(filename)

        # Update file status in Supabase
        self.supabase.table("files").update({"status": "processing"}).eq("file_id", file_id).execute()

        # Generate embeddings from the images
        BATCH_SZ = 4
        embeddings_data = []
        batches = [images[i:i + BATCH_SZ] for i in range(0, len(images), BATCH_SZ)]

        for batch_idx, batch in enumerate(batches):
            batch_images = self.colqwen2_processor.process_images(batch).to(
                self.colqwen2_model.device
            )
            batch_embeddings = list(self.colqwen2_model(**batch_images).to("cpu"))

            # Prepare data for Supabase insertion
            start_page = batch_idx * BATCH_SZ
            for i, embedding in enumerate(batch_embeddings):
                page_number = start_page + i
                if page_number < len(images):
                    # Use the standardized processing function
                    vector = self._process_embedding_tensor(embedding)

                    embeddings_data.append({
                        "file_id": file_id,
                        "page_number": page_number,
                        "vector": vector,
                        "model": "colqwen2-v0.1"
                    })
        print(f"Generated {len(embeddings_data)} embeddings for file {file_id}")

        for embedding_data in embeddings_data:
            try:
                self.supabase.table("embeddings").insert(embedding_data).execute()
            except Exception as e:
                print(f"Failed to insert embedding for page {embedding_data['page_number']}: {str(e)}")

        # Update file status to completed
        self.supabase.table("files").update({
            "status": "indexed",
            "page_count": len(images),
            "last_updated": "now()"
        }).eq("file_id", file_id).execute()

        return {
            "file_id": file_id,
            "status": "success",
            "pages_processed": len(images),
            "embeddings_created": len(embeddings_data)
        }

    @modal.method()
    def search(self, query: str, limit: int = 5):
        """Search across all indexed PDFs for the given query"""
        with model_image.imports():
            import torch
            import numpy as np
            from PIL import Image

        print(f"Executing search for query: '{query}'")

        # Process the query using ColQwen2 (same model as for image embeddings)
        batch_queries = self.colqwen2_processor.process_queries([query]).to(
            self.colqwen2_model.device
        )
        query_embeddings = self.colqwen2_model(**batch_queries)
        print(f"Generated query embedding")

        # Process the query embedding using the same standardized function
        try:
            query_vector = self._process_embedding_tensor(query_embeddings[0])
            print(f"Query vector processed (dimension: {len(query_vector)})")
        except Exception as e:
            print(f"Error processing query vector: {str(e)}")
            raise ValueError(f"Failed to process query embedding: {str(e)}")

        try:
            # Search in Supabase using vector similarity
            print(f"Executing vector search in database (limit: {limit})")
            results = self.supabase.rpc(
                "search_embeddings",
                {
                    "query_vector": query_vector,
                    "match_limit": limit
                }
            ).execute()

            # Process and structure results
            search_results = []
            for item in results.data:
                file_id = item["file_id"]
                page_number = item["page_number"]
                score = item["similarity"]

                # Get image for content preview (optional)
                try:
                    image_path = PDF_ROOT / file_id / f"{str(page_number).zfill(3)}.jpg"
                    if image_path.exists():
                        # Here you could generate a text preview from the image using the VLM
                        # but for simplicity we'll just note that the preview is available
                        content_preview = "Page content available for preview"
                    else:
                        content_preview = None
                except Exception as e:
                    content_preview = None

                search_results.append({
                    "file_id": file_id,
                    "page_number": page_number,
                    "score": score,
                    "content_preview": content_preview
                })

            print(f"Search returned {len(search_results)} results")
            return {"results": search_results}
        except Exception as e:
            print(f"Search error: {str(e)}")
            # Return empty results on error
            return {"results": [], "error": str(e)}

    @modal.method()
    def generate_content_preview(self, file_id: str, page_number: int):
        """Generate text preview for a specific page using the Qwen2-VL model"""
        with model_image.imports():
            import torch
            from PIL import Image

        try:
            # Load image from storage
            image_path = PDF_ROOT / file_id / f"{str(page_number).zfill(3)}.jpg"
            if not image_path.exists():
                return {"error": "Image not found"}

            image = Image.open(image_path)

            # Generate content description using Qwen2-VL with improved prompt
            text_prompt = "Extract and summarize all the text content from this document page."

            # Use the correct parameter names
            inputs = self.qwen2_vl_processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            ).to("cuda:0")

            with torch.no_grad():
                generated_ids = self.qwen2_vl_model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.3
                )

            # Use batch_decode instead of decode
            response = self.qwen2_vl_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Extract only the response part after the prompt
            response = response.replace(text_prompt, "").strip()

            return {"preview": response}
        except Exception as e:
            print(f"Error generating preview: {str(e)}")
            return {"error": str(e)}

    @modal.method()
    def answer_question(self, query: str, limit: int = 5):
        """Search for relevant pages and generate an answer to the query"""
        with model_image.imports():
            import torch
            import numpy as np
            from PIL import Image

        print(f"Executing search for query: '{query}'")

        # Process the query using ColQwen2 (same model as for image embeddings)
        batch_queries = self.colqwen2_processor.process_queries([query]).to(
            self.colqwen2_model.device
        )
        query_embeddings = self.colqwen2_model(**batch_queries)
        print(f"Generated query embedding")

        # Process the query embedding using the same standardized function
        try:
            query_vector = self._process_embedding_tensor(query_embeddings[0])
            print(f"Query vector processed (dimension: {len(query_vector)})")
        except Exception as e:
            print(f"Error processing query vector: {str(e)}")
            raise ValueError(f"Failed to process query embedding: {str(e)}")

        try:
            # Search in Supabase using vector similarity
            print(f"Executing vector search in database (limit: {limit})")
            results = self.supabase.rpc(
                "search_embeddings",
                {
                    "query_vector": query_vector,
                    "match_limit": limit
                }
            ).execute()

            # Process and structure results
            search_results = {"results": []}
            context_texts = []

            for item in results.data:
                file_id = item["file_id"]
                page_number = item["page_number"]
                score = item["similarity"]

                # Get image for content preview
                try:
                    image_path = PDF_ROOT / file_id / f"{str(page_number).zfill(3)}.jpg"
                    if image_path.exists():
                        # Load image directly
                        image = Image.open(image_path)

                        # Generate content description using Qwen2-VL
                        # Make sure we're passing the image correctly to the processor
                        inputs = self.qwen2_vl_processor(
                            text="Extract and summarize all the text content from this document page.",
                            images=image,
                            return_tensors="pt"
                        ).to("cuda:0")

                        with torch.no_grad():
                            generated_ids = self.qwen2_vl_model.generate(
                                **inputs,
                                max_new_tokens=500,
                                do_sample=True,
                                temperature=0.3
                            )

                        text_content = self.qwen2_vl_processor.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )[0].replace("Extract and summarize all the text content from this document page.", "").strip()

                        context_texts.append(f"[Source: Document {file_id}, Page {page_number}]\n{text_content}")

                        search_results["results"].append({
                            "file_id": file_id,
                            "page_number": page_number,
                            "score": score,
                            "content_preview": "Page content available for preview",
                            "text_content": text_content
                        })
                    else:
                        search_results["results"].append({
                            "file_id": file_id,
                            "page_number": page_number,
                            "score": score,
                            "content_preview": None,
                            "text_content": None
                        })
                except Exception as e:
                    print(f"Error processing page {page_number}: {str(e)}")
                    search_results["results"].append({
                        "file_id": file_id,
                        "page_number": page_number,
                        "score": score,
                        "content_preview": None,
                        "text_content": f"Error: {str(e)}"
                    })

            # Now generate the answer using all context texts
            combined_context = "\n\n".join(context_texts) if context_texts else "No relevant context found."

            # Create prompt for generating the answer
            answer_prompt = f"""Based on the following context from document pages, please answer the question accurately and concisely:

    Question: {query}

    Context:
    {combined_context}

    Answer:"""

            # For text-only processing, don't include images parameter
            text_inputs = self.qwen2_vl_processor(
                text=answer_prompt,
                return_tensors="pt"
            ).to("cuda:0")

            with torch.no_grad():
                text_generated_ids = self.qwen2_vl_model.generate(
                    **text_inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7
                )

            full_response = self.qwen2_vl_processor.batch_decode(
                text_generated_ids, skip_special_tokens=True
            )[0]

            # Extract just the answer part
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response

            # Return both the answer and the search results for context
            return {
                "answer": answer,
                "sources": search_results
            }

        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": {"results": []}
            }

@app.function(
    image=model_image,
    secrets=[supabase_secret],
    volumes={"/vol/pdfs/": pdf_volume},
)
@modal.asgi_app()
def api_server():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    import io
    import os
    from supabase import create_client
    import uuid

    api = FastAPI(title="ColQwen-Search Service")
    model = ColQwenModel()

    # Initialize Supabase client
    supabase = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY")
    )

    @api.get("/")
    def service_status():
        """Service health check endpoint"""
        return {"status": "ok", "message": "Service is running"}

    @api.post("/api/v1/search")
    async def search_endpoint(query: SearchQuery):
        """Execute search across all indexed documents"""
        try:
            results = model.search.remote(query.query, query.limit)

            # Enhance results with text content
            if "results" in results:
                for result in results["results"]:
                    if result.get("content_preview") == "Page content available for preview":
                        try:
                            preview = model.generate_content_preview.remote(
                                result["file_id"],
                                result["page_number"]
                            )
                            if "preview" in preview:
                                result["text_content"] = preview["preview"]
                            else:
                                result["text_content"] = "Error generating text content"
                        except Exception:
                            result["text_content"] = "Error generating text content"

            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    @api.post("/api/v1/question")
    async def question_answering_endpoint(query: SearchQuery):
        """Answer questions about indexed documents"""
        try:
            result = model.answer_question.remote(query.query, query.limit)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Question answering error: {str(e)}")

    @api.get("/api/v1/preview/{file_id}/{page_number}")
    async def get_content_preview(file_id: str, page_number: int):
        """Get content preview for a specific page"""
        try:
            preview = model.generate_content_preview.remote(file_id, page_number)
            return preview
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preview generation error: {str(e)}")

    @api.post("/api/v1/embeddings/file")
    async def generate_embeddings(
            file: UploadFile = File(...),
    ):
        """Generate embeddings for a single uploaded PDF file"""
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        try:
            # Read file content
            file_content = await file.read()
            file_id = str(uuid.uuid4())

            # Upload file to Supabase Storage
            file_path = f"pdfs/{file_id}.pdf"
            print(f"Uploading file to path: {file_path}")

            try:
                supabase.storage.from_("documents").upload(
                    file_path,
                    file_content
                )
                print("File uploaded to storage successfully")
            except Exception as storage_error:
                print(f"Storage upload error: {str(storage_error)}")
                raise HTTPException(status_code=500, detail=f"Storage error: {str(storage_error)}")

            # Create file record in the database
            try:
                supabase.table("files").insert({
                    "file_id": file_id,
                    "filename": file.filename,
                    "size": len(file_content),
                    "storage_path": file_path,
                    "status": "pending"
                }).execute()
                print("File record added to database successfully")
            except Exception as db_error:
                print(f"Database insert error: {str(db_error)}")
                # Try to clean up storage if database insert fails
                supabase.storage.from_("documents").remove([file_path])
                raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

            # Generate embeddings asynchronously
            try:
                result = model.generate_embeddings_for_file.remote(file_id, file_content)
                print(f"Spawned embedding generation task: {result}")
                return JSONResponse(
                    status_code=202,
                    content={
                        "file_id": file_id,
                        "status": "processing",
                        "message": "Embedding generation started"
                    }
                )
            except Exception as spawn_error:
                print(f"Task spawn error: {str(spawn_error)}")
                raise HTTPException(status_code=500, detail=f"Task scheduling error: {str(spawn_error)}")

        except Exception as e:
            print(f"General error in generate_embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    @api.post("/api/v1/embeddings/files")
    async def generate_embeddings_multiple(
            files: List[UploadFile] = File(...)
    ):
        """Generate embeddings for multiple uploaded PDF files"""
        results = []

        for file in files:
            if not file.filename.endswith('.pdf'):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Only PDF files are supported"
                })
                continue

            try:
                # Read file content
                file_content = await file.read()
                file_id = str(uuid.uuid4())

                # Upload file to Supabase Storage
                file_path = f"pdfs/{file_id}.pdf"
                supabase.storage.from_("documents").upload(
                    file_path,
                    file_content
                )

                # Create file record in the database
                supabase.table("files").insert({
                    "file_id": file_id,
                    "filename": file.filename,
                    "size": len(file_content),
                    "storage_path": file_path,
                    "status": "pending"
                }).execute()

                # Generate embeddings asynchronously
                model.generate_embeddings_for_file.spawn(file_id, file_content)

                results.append({
                    "file_id": file_id,
                    "status": "processing",
                    "message": "Embedding generation started"
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Error: {str(e)}"
                })
        return {"results": results}

    @api.get("/api/v1/files")
    async def list_files():
        """List all indexed files"""
        try:
            response = supabase.table("files").select("*").execute()
            files = []
            for file in response.data:
                files.append({
                    "file_id": file["file_id"],
                    "filename": file["filename"],
                    "size": file["size"],
                    "created_at": file["created_at"],
                    "status": file["status"]
                })
            return {"files": files}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    @api.get("/api/v1/files/{file_id}")
    async def get_file_details(file_id: str):
        """Get details for a specific file"""
        try:
            # Get file info
            file_response = supabase.table("files").select("*").eq("file_id", file_id).execute()

            if not file_response.data:
                raise HTTPException(status_code=404, detail="File not found")

            file = file_response.data[0]

            # Count embeddings for this file
            embedding_count_response = supabase.table("embeddings") \
                .select("*", count="exact") \
                .eq("file_id", file_id) \
                .execute()

            embedding_count = embedding_count_response.count

            return {
                "file_id": file["file_id"],
                "filename": file["filename"],
                "size": file["size"],
                "created_at": file["created_at"],
                "status": file["status"],
                "embedding_count": embedding_count,
                "last_updated": file.get("last_updated", file["created_at"])
            }
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    @api.delete("/api/v1/files/{file_id}")
    async def delete_file(file_id: str):
        """Delete a file and its embeddings"""
        try:
            # Check if file exists
            file_response = supabase.table("files").select("storage_path").eq("file_id", file_id).execute()

            if not file_response.data:
                raise HTTPException(status_code=404, detail="File not found")

            storage_path = file_response.data[0]["storage_path"]

            # Delete embeddings
            supabase.table("embeddings").delete().eq("file_id", file_id).execute()

            # Delete file record
            supabase.table("files").delete().eq("file_id", file_id).execute()

            # Delete file from storage
            supabase.storage.from_("documents").remove([storage_path])

            # Clean up the PDF images from volume
            import shutil
            pdf_dir = PDF_ROOT / file_id
            if pdf_dir.exists():
                shutil.rmtree(pdf_dir)

            return {"status": "success", "message": f"File {file_id} deleted successfully"}
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")

    return api


# Deploy instructions
if __name__ == "__main__":
    print("To deploy this service, run: modal deploy colqwen_search_service.py")
    print("Make sure you've set up Supabase and created the necessary secrets.")