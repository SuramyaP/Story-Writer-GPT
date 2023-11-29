import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = apikey

#APP BEGINS HERE

st.title('Story Writer GPT')
prompt = st.text_input('Enter your Prompt')

#Prompt templates

title_template = PromptTemplate(
	input_variables = ['topic']
	template = 'Write me a story title about {topic}'
	
	)

script_template = PromptTemplate(
	input_variables = ['topic']
	template = 'Write me a story script based on {topic}'
	
	)

#Memory

memory = ConversationBufferMemory(input_keu = 'topic', memory_key = 'chat_history')



#LLM BEgins here

llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm=llm, prompt = title_template, verbose = True, output_key ='title', memory=memory)
script_chain = LLMChain(llm=llm, prompt = script_template, verbose = True, output_key ='script', memory=memory)
sequential_chain = SequentialChain(chains = [title_chain,script_chain], input_variables=['topic'], output_variables=['title','script'], verbose = True, memory=memory )




if prompt:
	response = sequential_chain({'topic':prompt})
	st.write (response['title'])
	st.write (response['script'])


	with st.expander('Message History'):
		st.info(memory.buffer)









