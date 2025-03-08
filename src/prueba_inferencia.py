from conette import CoNeTTEConfig, CoNeTTEModel

config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)

path = "src/conette/data/sample.wav"
outputs = model(path)
candidate = outputs["cands"][0]
print(candidate)