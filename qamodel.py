from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def generate_answer(question: str, context: list) -> str:
    """
    Generate an answer based on a given question and a list of documents.

    Args:
        question (str): The question to be answered.
        context (list): A list of documents to search for the answer.

    Returns:
        str: The generated answer.
    """
    # Combine the documents into a single context string
    full_context = " ".join(context)

    # Tokenize and generate
    inputs = tokenizer(question, full_context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most likely start and end of the answer
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)

    # Get the answer tokens and decode them
    question = question.lower()
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens)
    if answer == "<s>" and question == "hi":
        answer = "Hi how can I help you today?"
    elif answer == "<s>" and question == "bye":
        answer = "Bye! Have a great day!"
    elif answer == "<s>" and question == "thanks":
        answer = "You're welcome! If you have any other questions, feel free to ask."
    elif answer == "<s>" and question == "hello":
        answer = "Hello! How can I help you today?"
    elif answer == "<s>" and question == "goodbye":
        answer = "Goodbye! Have a great day!"
    elif answer == "<s>" and question == "thank you":
        answer = "You're welcome! If you have any other questions, feel free to ask."
    elif answer == "<s>" and question == "good morning":
        answer = "Good morning! How can I help you today?"
    elif answer == "<s>" and question == "good afternoon":
        answer = "Good afternoon! How can I help you today?"
    elif answer == "<s>":
        answer = "I don't know, I will ask for a human agent to help you."
    
    
    # print("--------------------------------")
    # print("Question:", question)
    # print("Answer:", answer)

    return answer





#test code
# folder_path = "testdata"

# documents = []
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         with open(os.path.join(folder_path, filename), "r") as f:
#             documents.append(f.read())

# print(f"Loaded {len(documents)} documents.")


# user_question = input("Please enter your question: ")
# while user_question != "exit":
#     answer = generate_answer(user_question, documents)
#     print("Answer:", answer)
#     user_question = input("Please enter your question: ")

# print("Exiting...")