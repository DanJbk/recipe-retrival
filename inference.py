from transformers import AutoTokenizer
from url_fetch import html_scanner
import sys, getopt
import numpy as np
import json
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = torch.load("model.pth", map_location=device)


def predict(model, text, tokenizer=tokenizer):

    model.eval()
    model = model.to(device)
    batch = tokenizer([text], padding="max_length", truncation=True)
    batch = {k: torch.tensor(v).to(device).type(torch.long) for k, v in batch.items()}
    outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    return predictions.cpu().item(), logits


def main(argv):

    test_url = getopt.getopt(argv, "")[1][0]

    result = {"INSTRUCTIONS": "", "Recipe": ""}

    instructions = [[], []]
    ingridients = [[], []]

    candidates = html_scanner.fetch_text(test_url)
    for i, text in enumerate(candidates):
        prediction, logists = predict(model, text)

        if prediction != 0:
            if prediction == 1:
                instructions[1].append(torch.softmax(logists, 1).max())
                instructions[0].append(text)
            elif prediction == 2:
                ingridients[1].append(torch.softmax(logists, 1).max())
                ingridients[0].append(text)

    best_match = torch.argmax(torch.tensor(ingridients[1]))
    result["Recipe"] = ingridients[0][best_match]

    best_match = torch.argmax(torch.tensor(instructions[1]))
    result["INSTRUCTIONS"] = instructions[0][best_match]

    with open("output.json", "w") as outfile:
        json.dump(result, outfile)


if __name__ == "__main__":
    main(sys.argv[1:])
