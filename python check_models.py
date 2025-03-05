import google.generativeai as genai

genai.configure(api_key="AIzaSyBt9DFm2ZmPXyBp7gdG3tvk9MPZM3LABys")  # Replace with your API key

models = genai.list_models()
for model in models:
    print(model.name)
