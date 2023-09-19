import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def read_lines_from_file(file_path):
    #Read lines from a file using its file path
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def predict_labels_for_sentences(model, tokenizer, sentences, index_to_label_dict):
    #Predict labels using a model and write the predictions into a file.
    input_tensors = tokenizer(sentences, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    outputs = model(**input_tensors)
    logit_values = outputs.logits
    predicted_labels_for_all_sents = []
    with torch.no_grad():
        softmax_layer = torch.nn.Softmax(dim=1)
        output_predicted_probs_torch = softmax_layer(logit_values)
        arg_max_torch = torch.argmax(output_predicted_probs_torch, axis=-1)
        arg_max_torch = arg_max_torch.tolist()

        for index, sentence in enumerate(sentences):
            word_ids = input_tensors.word_ids(batch_index=index)
            previous_word_idx = None
            label_ids = []

            for word_index in range(len(word_ids)):
                # Special tokens have a word id that is None. We set the label to -100, so they are automatically
                # ignored in the loss function.
                if word_ids[word_index] is None:
                    continue
                # We set the label for the first token of each word.
                elif word_ids[word_index] != previous_word_idx:
                    label_ids.append(index_to_label_dict[arg_max_torch[index][word_index]])
                # For the other tokens in a word, we ignore the label prediction
                else:
                    continue
                previous_word_idx = word_ids[word_index]

            tokens_with_preds = [token + '\t' + pred_label for token, pred_label in zip(sentence.split(' '), label_ids)]

            predicted_labels_for_all_sents.append('\n'.join(tokens_with_preds) + '\n')
    return predicted_labels_for_all_sents


def write_lines_to_file(lines, file_path):
    #Write lines to a file.
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def main():
    #Pass arguments and call functions here.
    input_data = """हैलो दोस्त। एनएलपी की दुनिया में आपका स्वागत है"""

    tokenizer = AutoTokenizer.from_pretrained("Your_Model_here")
    model = AutoModelForTokenClassification.from_pretrained("Your_Model_here")
    label_file = "labels_list.txt"
    labels = read_lines_from_file(label_file)
    index_to_label_dict = {index: label for index, label in enumerate(labels)}

    sentences = input_data.split('\n')  # Split the input data into sentences
    predicted_labels_for_all_sents = predict_labels_for_sentences(model, tokenizer, sentences, index_to_label_dict)

    output_folder = "output_naya.txt"
    write_lines_to_file(predicted_labels_for_all_sents, output_folder)

if __name__ == '__main__':
    main()
