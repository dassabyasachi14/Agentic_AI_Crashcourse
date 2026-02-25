from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import List

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


class Recipe(BaseModel):
    name: str = Field(description="Name of the dish")
    ingredients: List[str] = Field(description="List of required ingredients")
    prep_time_minutes: int = Field(description="Time to prepare in minutes")

parser = PydanticOutputParser(pydantic_object=Recipe)


#template= ChatPromptTemplate.from_messages([("system", "You are a helpful assistant"),
#                                   ("user","Find the name, ingredients and preparation time from the recipe provided. Make sure to use only the input provided \n\nRecipe:{recipe}")])

template=PromptTemplate.from_template("Find the name, ingredients and preparation time from the recipe provided. Make sure to use only the input provided \n\nRecipe:{recipe} \n Format Instructions: {format_instructions}")

partial_template=template.partial(format_instructions=parser.get_format_instructions())

recipe_text = "To make tea, use water, milk and tea leaves and boil for 15 mins"

#print(partial_template.format(recipe=recipe_text))


llm=ChatOpenAI(model= "gpt-4o-mini", temperature=0)

chain= partial_template|llm|parser

recipe_text = "To make tea, use water, milk and tea leaves and boil for 15 mins"

output = chain.invoke({"recipe":recipe_text})
print(output)

#for chunk in chain.stream({"recipe":recipe_text}):
#    print(chunk)