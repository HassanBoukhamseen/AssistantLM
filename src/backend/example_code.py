

# for part in stream:
#     print(part.choices[0].delta.content or " ", end="")

# print(" ")
# prompt = "what are the statisitcs regarding the number of emergency calls the UAE gets yearly?"

# stream = client.chat.completions.create(
#     model=,
#      messages=[
#             {"role": "system", "content": "You will be given a prompt and context information to use in answering it. The prompts will mainly concern governmental processes of the United Arab Emirates. You will not use your background knowledge, and only reference the context given in answering prompts."},
#             {"role": "user", "content": prompt}
#         ],
#     stream=True,
# )
# for part in stream:
#     print(part.choices[0].delta.content or " ", end="")




# for event in replicate.stream(
#     "meta/llama-2-70b-chat",
#     input=input
# ):
#     print(event, end="")

# print("\n===============================")
# input = {
#     "prompt": "explain what machine learning is",
#     "temperature": 1
# }

# output = replicate.run(
#     "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
#     input=input
# )
# print("".join(output))
   