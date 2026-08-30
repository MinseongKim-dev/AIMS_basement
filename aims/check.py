from open_clip import create_model_from_pretrained, get_tokenizer

model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)

print("BiomedCLIP 로드 완료")
print(f"model type: {type(model)}")