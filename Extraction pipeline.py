
import subprocess
import sys

def setup_dependencies():
    print("Checking and installing packages from requirements.py...")
    subprocess.check_call([sys.executable, "requirements.py"]) # Runs requirements.py as an isolated background system process
    print("Setup complete!")

setup_dependencies()


class ProductAnalysisPipeline:
    """
    Multi-LLM extraction pipeline with ensemble approach
    Uses 3 text LLMs + 3 vision models + synthesis LLM
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")

        # Comprehensive extraction prompt template
        self.extraction_prompt = """Extract ALL available product information from the text below. Be thorough and precise.

        Required fields (extract if present):
        1. Brand Name
        2. Product Name
        3. Product Category/Type
        4. Ingredient Claims (organic, natural, gluten-free, vegan, non-GMO, etc.)
        5. Certifications (USDA Organic, Fair Trade, Non-GMO Project, etc.)
        6. Key Features/Benefits
        7. Allergen Information
        8. Any other important details for price prediction

        Product Text: {text}

        Extracted Information:"""

        # Will store embeddings
        self.embedding_model = None
        self.embedding_tokenizer = None

    def _clear_memory(self):
        """Clear GPU memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def save_cache(self, embeddings: np.ndarray, labels: np.ndarray, filename: str = 'embeddings_cache.npz'):
        """Saves embeddings and target labels to a compressed NumPy file."""
        print(f"\n✓ Saving {len(embeddings)} embeddings and labels to '{filename}'...")
        np.savez_compressed(filename, embeddings=embeddings, labels=labels)
        print("Save complete.")

    def load_cache(self, filename: str = 'embeddings_cache.npz') -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Loads embeddings and target labels from a compressed NumPy file."""
        if os.path.exists(filename):
            print(f"\n✓ Loading cached embeddings and labels from '{filename}'...")
            cache = np.load(filename)
            embeddings = cache['embeddings']
            labels = cache['labels']
            print(f"Loaded {len(embeddings)} items.")
            return embeddings, labels
        else:
            print(f"\n❌ Cache file '{filename}' not found. Will generate new embeddings.")
            return None, None

    # TEXT EXTRACTION 

    def extract_with_phi2(self, product_text: str) -> str:
        """Full extraction with Phi-2"""
        print("    Loading Phi-2...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map="auto",
            torch_dtype=torch.bfloat16 # Use bfloat16 for better numerical stability if GPU supports it
        )

        # 3. Load the tokenizer normally
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

        prompt = self.extraction_prompt.format(text=product_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

        del model, tokenizer, inputs, outputs
        self._clear_memory()
        return result

    def extract_with_tinyllama(self, product_text: str) -> str:
        """Full extraction with TinyLlama"""
        print("    Loading TinyLlama...")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

        messages = [
            {
                "role": "system",
                "content": "You are an expert at extracting comprehensive product information. Extract all details thoroughly."
            },
            {
                "role": "user",
                "content": self.extraction_prompt.format(text=product_text)
            }
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in result:
            result = result.split("<|assistant|>")[-1].strip()

        del model, tokenizer, inputs, outputs
        self._clear_memory()
        return result

    def extract_with_flan(self, product_text: str) -> str:
        """Full extraction with Flan-T5"""
        print("    Loading Flan-T5...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            torch_dtype=torch.float16
        ).to(self.device)

        prompt = self.extraction_prompt.format(text=product_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True)

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model, tokenizer, inputs, outputs
        self._clear_memory()
        return result

    # IMAGE EXTRACTION 
    def get_image(self, url):

        MAX_RETRIES = 10
        RETRY_DELAY = 2

        print(f"Attempting to download image from: {url}")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                image_data = BytesIO(response.content)
                image = Image.open(image_data).convert("RGB")

                print(f"Image successfully downloaded and opened on attempt {attempt + 1}. Size: {image.size}")
                return image

            except (requests.exceptions.RequestException, Image.UnidentifiedImageError) as e:
                if attempt < MAX_RETRIES - 1:
                    # Log failure and pause before the next attempt
                    print(f"Attempt {attempt + 1} failed ({type(e).__name__}). Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    # Max retries reached
                    print(f"All {MAX_RETRIES} attempts failed. Final error: {e}. Returning None.")
                    return None

    def extract_from_image_smolvlm(self, image) -> str:
        """Full text extraction from image using SmolVLM"""
        print("    Loading SmolVLM...")
        # In extract_from_image_smolvlm:

        model_id = "HuggingFaceTB/SmolVLM2-256M-Instruct"

        processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Instruct",
            trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Instruct",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        ).to(self.device)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": """Extract ALL visible text and information from this product image:
                    - Brand name
                    - Product name
                    - Size/quantity
                    - Ingredient claims (organic, vegan, etc.)
                    - Certifications
                    - Any other visible text or claims
                    Be thorough and extract everything you can see."""}
            ]
        }]

        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            messages,  # Pass the messages list directly
            images=image,
            return_tensors="pt",
            add_generation_prompt=True, # Include this if the processor needs it
            tokenize=True
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

        result = processor.decode(outputs[0], skip_special_tokens=True)

        del model, processor, inputs, outputs
        self._clear_memory()
        return result

    def extract_from_image_trocr(self, image) -> str:
        """OCR extraction using TrOCR"""
        print("    Loading TrOCR...")
        processor = AutoProcessor.from_pretrained("microsoft/trocr-large-printed", use_fast=True)
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(self.device)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=200)

        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        del model, processor, pixel_values, generated_ids
        self._clear_memory()
        return result

    def extract_from_image_blip(self, image) -> str:
        """Visual understanding using BLIP"""
        print("    Loading BLIP...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        inputs = processor(image, text="""You are a visual context analyst. Your task is to analyze the product image and provide a structured, descriptive summary of its overall presentation, excluding any focus on fine print.

                INSTRUCTIONS:

                Focus on Visuals: Describe the product's aesthetic appearance, packaging, and the quality of the photograph itself.

                Structured Output: Provide a single, detailed response covering the four mandated categories below.

                CATEGORIES FOR VISUAL ANALYSIS:

                Identity & Category: State the primary product category (e.g., 'snack bar', 'shampoo') and the main brand name visible, ignoring small text.

                Packaging Aesthetics: Describe the physical container type (e.g., box, stand-up pouch, glass jar). Identify the two most dominant colors and describe the surface texture (e.g., matte, glossy, metallic).

                Design & Staging: Characterize the overall design aesthetic (e.g., minimalist, playful, vintage). Is the product photographed alone on a neutral background, or is it professionally staged with props or ingredients?

                Image Quality Assessment: Assess the technical quality of the photograph. State whether the image is sharp, well-lit, and professionally composed, or if it is blurry, dark, low-resolution, or poorly cropped.""",
                          return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150)

        result = processor.decode(out[0], skip_special_tokens=True)

        del model, processor, inputs, out
        self._clear_memory()
        return result

    # SYNTHESIS

    def synthesize_with_validation(self, text_extracts: dict, image_extracts: dict = None) -> dict:
        """Intelligent synthesis with cross-validation and confidence scoring"""
        print("    Loading Qwen2.5-3B for synthesis...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

        prompt = f""" You are the final Data Fusion and Structured Output Engine for a product intelligence pipeline. Your sole function is to ingest multiple VLP model outputs, cross-validate them, resolve conflicts, and output the result as a single, valid JSON object, and nothing else.
                INSTRUCTIONS:

                Data Validation: Analyze the inputs from all six models (TrOCR, SmolVLM, BLIP, Phi-2, TinyLlama, Flan-T5).

                Conflict Resolution: Use the most specific text (TrOCR/SmolVLM) to validate factual fields (Brand, Claims). Use the visual information (BLIP) for aesthetic and quality fields.

                Confidence Scoring: Assign a final confidence score (between 0.0 and 1.0) based on consensus and data clarity.

                MANDATORY OUTPUT FORMAT:
                You MUST output ONLY a single, well-formed JSON object that adheres to the following schema. Do not include any text, reasoning, or markdown outside of the JSON block.

                final_brand_name (string): The single most agreed-upon brand name.

                product_category (string): The refined, specific category (e.g., "protein bar," "facial moisturizer").

                size_quantity (string): The quantity or weight (e.g., "12 oz," "100g," "24 count").

                key_claims (array of strings): All unique ingredient or health claims (e.g., ["Organic", "Vegan", "Non-GMO"]).

                certifications_found (array of strings): All verifiable certifications or logos (e.g., ["USDA Organic", "Cruelty-Free"]).

                visual_aesthetic (string): The high-level design style (e.g., "Minimalist," "Playful," "Vintage").

                packaging_type (string): The physical container (e.g., "Glass Bottle," "Stand-up Pouch," "Cardboard Box").

                image_technical_quality (string): Assessment of the photo quality (e.g., "Sharp and Professional," "Slightly Blurry," "Low Resolution").

                fusion_confidence_score (float): The overall confidence in the final extracted data (0.0 to 1.0).


                ---TEXT MODELS---

                MODEL 1 (Phi-2):
                {text_extracts.get('phi2', 'N/A')}

                MODEL 2 (TinyLlama):
                {text_extracts.get('tinyllama', 'N/A')}

                MODEL 3 (Flan-T5):
                {text_extracts.get('flan', 'N/A')}
                """
        if image_extracts:
          prompt += f"""

                ---IMAGE MODELS---

                MODEL 4 (SmolVLM):
                {image_extracts.get('smolvlm', 'N/A')}

                MODEL 5 (TrOCR):
                {image_extracts.get('trocr', 'N/A')}

                MODEL 6 (BLIP):
                {image_extracts.get('blip', 'N/A')}
                """

        prompt += "\n\nValidated and Synthesized Output:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.2,
                do_sample=True,
                top_p=0.9
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

        del model, tokenizer, inputs, outputs
        self._clear_memory()

        # Parse confidence scores

        return {
            'synthesis': result,
            'raw_extracts': {
                'text': text_extracts,
                'image': image_extracts
            }
        }


    # EMBEDDING CREATION

    def create_dense_representation(self, text: str) -> np.ndarray:
        """Convert text to dense embedding using sentence-transformers"""
        if self.embedding_model is None:
            print("    Loading embedding model...")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

        inputs = self.embedding_tokenizer(text, return_tensors='pt', truncation=True,
                                          max_length=512, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        return embeddings.cpu().numpy().flatten()

    # MAIN PROCESSING

    def process_row(self, row, text_column='catalog_content', image_column=None):
        """Process a single row with full ensemble extraction"""
        product_text = str(row[text_column])
        match = re.search(r'(.+?)Value:', product_text)

        if match:
          # If 'Value:' is found, use only the part before it (stripped of whitespace)
          product_text = match.group(1).strip()
        else:
          product_text = product_text

        # Extract from image if available
        image_extracts = None
        if image_column and pd.notna(row[image_column]):
            print("  [IMAGE EXTRACTION - Full Ensemble]")
            try:
                img = self.get_image(row[image_column])
                img_smolvlm = self.extract_from_image_smolvlm(img)
                img_trocr = self.extract_from_image_trocr(img)
                img_blip = self.extract_from_image_blip(img)

                image_extracts = {
                    'smolvlm': img_smolvlm,
                    'trocr': img_trocr,
                    'blip': img_blip
                }
            except Exception as e:
                print(f"  ⚠ Image processing failed: {e}")

        print("  [TEXT EXTRACTION - Full Ensemble]")
        text_phi2 = self.extract_with_phi2(product_text)
        text_tinyllama = self.extract_with_tinyllama(product_text)
        text_flan = self.extract_with_flan(product_text)

        text_extracts = {
            'phi2': text_phi2,
            'tinyllama': text_tinyllama,
            'flan': text_flan
        }



        # Synthesis
        print("  [SYNTHESIS & VALIDATION]")
        synthesis_result = self.synthesize_with_validation(text_extracts, image_extracts)

        # Create embedding
        print("  [EMBEDDING CREATION]")
        embedding = self.create_dense_representation(synthesis_result['synthesis'])

        return {
            'synthesis': synthesis_result['synthesis'],
            'embedding': embedding,
            'raw_extracts': synthesis_result['raw_extracts']
        }

    def process_dataframe(self, df, text_column='catalog_content', image_column=None,
                      price_column='price', save_raw_extracts=False, cache_filename='embeddings_cache.npy'):
        """Process entire dataframe, checking for and saving embeddings cache."""

        # Cache Check 
        # Attempt to load from cache first
        cached_embeddings = self.load_cache(cache_filename)

        if cached_embeddings is not None and len(cached_embeddings) == len(df):
            print("Using cached embeddings. Skipping LLM/Vision pipeline.")
            # If cached, we return the original DF and the loaded embeddings.

            # We need to fill the 'embedding' column in the returned dataframe
            df_results = df.copy()
            df_results['embedding'] = list(cached_embeddings)

            # Re-run the regex separation just in case (cheap operation)
            regex_pattern = r'^(.+?)Value:\s*(.+?)Unit:\s*(.+?)$'
            df_results[['Feature_Name', 'Value', 'Unit']] = df_results[text_column].str.extract(
                regex_pattern, expand=True
            ).apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            df_results[['Feature_Name', 'Value', 'Unit']] = df_results[['Feature_Name', 'Value', 'Unit']].fillna('')
            print("[REGEX SEPARATION] Complete.")

            return df_results, cached_embeddings
       

        # If cache is not found or size mismatch, proceed with generation
        results = []
        embeddings = []

        print(f"\n{'='*80}")
        print(f"PROCESSING {len(df)} ROWS WITH ENSEMBLE EXTRACTION (NO CACHE FOUND)")
        print(f"{'='*80}\n")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing products"):
            print(f"\n{'─'*80}")
            print(f"Row {idx + 1}/{len(df)}")
            print(f"{'─'*80}")
            try:
                result = self.process_row(row, text_column, image_column)
                results.append(result)
                embeddings.append(result['embedding'])

            except Exception as e:
                print(f"  ❌ Error processing row {idx}: {e}")
                import traceback
                traceback.print_exc()
                results.append(None)
                embeddings.append(np.zeros(384)) # Use 384 dimensions for all-MiniLM-L6-v2 embedding size

        # Create results dataframe
        df_results = df.copy()
        df_results['synthesis'] = [r['synthesis'] if r else '' for r in results]
        df_results['embedding'] = embeddings

        # Save the newly generated embeddings
        embeddings_array = np.array(embeddings)
        self.save_cache(embeddings_array, cache_filename)

        # Regex separation (as before)
        regex_pattern = r'^(.+?)Value:\s*(.+?)Unit:\s*(.+?)$'
        df_results[['Feature_Name', 'Value', 'Unit']] = df_results[text_column].str.extract(
            regex_pattern,
            expand=True
        ).apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        df_results[['Feature_Name', 'Value', 'Unit']] = df_results[['Feature_Name', 'Value', 'Unit']].fillna('')

        print("[REGEX SEPARATION] Complete. New columns 'Feature_Name', 'Value', 'Unit' created.")

        return df_results, embeddings_array

    # CLUSTERING

    def perform_clustering(self, embeddings, n_clusters=5):
        """Perform K-means clustering"""
        print("\n" + "="*80)
        print("PERFORMING K-MEANS CLUSTERING")
        print("="*80)

        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        print(f"\nFitting K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)

        print("\n" + "-"*80)
        print("CLUSTER DISTRIBUTION")
        print("-"*80)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"Cluster {cluster_id}: {count} products ({count/len(cluster_labels)*100:.1f}%)")

        return {
            'model': kmeans,
            'labels': cluster_labels,
            'scaler': scaler
        }

