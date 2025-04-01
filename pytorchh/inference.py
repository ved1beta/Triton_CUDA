# Required installations:
# pip install transformers torch sentencepiece sacremoses

from transformers import MarianMTModel, MarianTokenizer
import torch
import logging
from typing import Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_translation_model(
    source_lang: str,
    target_lang: str
) -> Tuple[Optional[MarianMTModel], Optional[MarianTokenizer], Optional[str]]:
    """
    Load a pre-trained translation model and tokenizer from Hugging Face.
    
    Args:
        source_lang (str): Source language code (e.g., 'en', 'fr', 'de')
        target_lang (str): Target language code (e.g., 'en', 'fr', 'de')
    
    Returns:
        Tuple containing (model, tokenizer, device) or (None, None, None) if loading fails
    """
    # Construct model name using language codes
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    
    try:
        # Load tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        return model, tokenizer, device
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Available language pairs can be found at: "
                   "https://huggingface.co/Helsinki-NLP")
        return None, None, None

def translate_text(
    text: str,
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    device: str,
    max_length: int = 128,
    num_beams: int = 4,
    length_penalty: float = 0.6
) -> Optional[str]:
    """
    Translate text using the loaded model and tokenizer
    
    Args:
        text (str): Text to translate
        model (MarianMTModel): Loaded translation model
        tokenizer (MarianTokenizer): Loaded tokenizer
        device (str): Device to use for computation ('cuda' or 'cpu')
        max_length (int): Maximum length of the generated translation
        num_beams (int): Number of beams for beam search
        length_penalty (float): Length penalty for generation
    
    Returns:
        str: Translated text, or None if translation fails
    """
    try:
        # Input validation
        if not text.strip():
            raise ValueError("Empty text provided for translation")
            
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        translated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True
        )
        
        # Decode the translated tokens
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return translated_text
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return None

def main():
    # Example: English to French translation
    source_lang = "en"
    target_lang = "fr"  # Changed from 'ja' to 'fr' as it's more commonly available
    
    # Load model
    logger.info(f"Loading translation model for {source_lang} to {target_lang}...")
    model, tokenizer, device = load_translation_model(source_lang, target_lang)
    
    if model is not None:
        # Example texts
        text = "Hello, how are you?",
           
        
        
        # Perform translations
        
        translation = translate_text(text, model, tokenizer, device)
           

if __name__ == "__main__":
    main()