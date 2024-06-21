## The following solution worked well for us

########################################################################################
## Task 2

agent.agent.llm_chain.prompt.template = """
<s>[INST/]<<SYS>>
Answer the following questions as best you can. You have access to the following tools:

[Python REPL]: A Python shell. Use this to execute python commands. Input should be a valid python command.
        If you expect output it should be printed out.

===================================
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of ["Python REPL"]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

===================================
Use the following as an example of how you should perform: 

Question: Please tell me 42 x 24
Thought: I can just use python to perform this operation
Action: Python REPL
Action-Input: 
def multiply_add(a, b, c):
    return a * b + c
print('42 x 24 + 12 is', multiply_add(42, 24, 12))
Observation: 42 x 24 + 12 is 1020
Thought: Based on the observation, I am ready to answer
Final Answer: the final answer is 1008

<</SYS>>
=================================
Question: {input}
[/INST]
Thought: {agent_scratchpad}
"""

####################################################################################
## TODO: Your workspace is below

llama_full_prompt = PromptTemplate.from_template(
    template="<s>[INST]<<SYS>>{sys_msg}<</SYS>>\n\nContext:\n{history}\n\nHuman: {input}\n[/INST] {primer}",
)

llama_prompt = llama_full_prompt.partial(
    sys_msg=(
        "You are a helpful, respectful and honest AI assistant."
        "\nAlways answer as helpfully as possible, while being safe."
        "\nPlease be brief and efficient unless asked to elaborate, and follow the conversation flow."
        "\nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
        "\nEnsure that your responses are socially unbiased and positive in nature."
        "\nIf a question does not make sense or is not factually coherent, explain why instead of answering something incorrect."
        "\nIf you don't know the answer to a question, please don't share false information."
        "\nIf the user asks for a format to output, please follow it as closely as possible."
    ),
    primer="",
    history="",
)

####################################################################################
## THESE MIGHT BE USEFUL IMPORTS!

from langchain.chains import ConversationChain
from glob import glob

img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
emo_pipe = pipeline('sentiment-analysis', 'SamLowe/roberta-base-go_emotions')
zsc_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
tox_pipe = pipeline("text-classification", model="nicholasKluge/ToxicityModel")


## WARNING: toxic_pipe returns the reward, where reward = 1 - toxicity

###################################################################################


class MyAgent(MyAgentBase):
    ## Instance methods that can be passed in as BaseModel arguments.
    ## Will be associated with self

    general_prompt: PromptTemplate
    llm: BaseLLM

    general_chain: Optional[LLMChain]
    max_messages: int = Field(10, gt=1)

    temperature: float = Field(0.6, gt=0, le=1)
    max_new_tokens: int = Field(128, ge=1, le=2048)
    eos_token_id: Union[int, List[int]] = Field(2, ge=0)
    gen_kw_keys = ['temperature', 'max_new_tokens', 'eos_token_id']
    gen_kw = {}

    user_toxicity: float = 0.5
    user_emotion: str = "Unknown"

    @root_validator
    def validate_input(cls, values: Any) -> Any:
        '''Think of this like the BaseModel's __init__ method'''
        if not values.get('general_chain'):
            llm = values.get('llm')
            prompt = values.get("general_prompt")
            memory = ConversationSummaryMemory(llm=llm, temperature=0, verbose=False)

            llama_template_hist = prompt.copy()
            llama_template_hist.input_variables = ['input', 'history']

            values['general_chain'] = ConversationChain(
                llm=llm,
                prompt=llama_template_hist,
                memory=memory,
                verbose=True
            )  # LLMChain(llm=llm, prompt=prompt)  ## <- Feature stop
        values['gen_kw'] = {k: v for k, v in values.items() if k in values.get('gen_kw_keys')}
        return values

    def process_img(self, inputs: List[str]):
        sentense = ''
        for inpt in inputs:
            sentense += inpt
        return sentense

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any):
        '''Takes in previous logic and generates the next action to take!'''

        ## [Base Case] Default message to start off the loop. DO NOT OVERRIDE
        tool, response = "Ask-For-Input Tool", "Hello World! How can I help you?"
        if len(intermediate_steps) == 0:
            return self.action(tool, response)

        ## History of past agent queries/observations
        queries = [step[0].tool_input for step in intermediate_steps]
        observations = [step[1] for step in intermediate_steps]
        last_obs = observations[-1]  # Most recent observation (i.e. user input)

        if last_obs:
            print(last_obs)
            imgs = last_obs.split('`')
            print(type(imgs))
            if len(imgs) > 0:
                last_obs = self.process_img(imgs)
                print(last_obs)
            self.user_emotion = emo_pipe(last_obs)[0]['label']
            self.user_toxicity = 1 - tox_pipe(last_obs)[0]['score']

        #############################################################################
        ## FOR THIS METHOD, ONLY MODIFY THE ENCLOSED REGION

        ## [!] Probably a good spot for your user statistics tracking

        ## [Stop Case] If the conversation is getting too long, wrap it up
        if len(observations) >= self.max_messages:
            response = "Thanks so much for the chat, and hope to see ya later! Goodbye!"
            return self.action(tool, response, finish=True)

        ## [!] Probably a good spot for your input-augmentation steps

        ## [Default Case] If observation is provided and you want to respond... do it!
        with SetParams(llm, **self.gen_kw):
            response = self.general_chain.run(last_obs)

        ## [!] Probably a good spot for your output-postprocessing steps

        ## FOR THIS METHOD, ONLY MODIFY THE ENCLOSED REGION
        #############################################################################

        ## [Default Case] Send over the response back to the user and get their input!
        return self.action(tool, response)

    def reset(self):
        self.user_toxicity = 0
        self.user_emotion = "Unknown"
        if getattr(self.general_chain, 'memory', None) is not None:
            self.general_chain.memory.clear()  ## Hint about what general_chain should be...


####################################################################################
## Define how you want your conversation to go. You can also use your own input
## The below example in conversation_gen exercises some of the requirements.

student_name = "Jacky Du"  ## TODO: What's your name
ask_via_input = False  ## TODO: When you're happy, try supplying your own inputs


def conversation_gen():
    yield f"Hello! How's it going? My name is {student_name}! Nice to meet you!"
    yield "Please tell me a little about deep learning!"
    yield "What's my name?"  ## Memory buffer
    yield "I'm not feeling very good -_-. What should I do"  ## Emotion sensor
    yield "No, I'm done talking! Thanks so much!"  ## Conversation ender
    yield "Goodbye!"  ## Conversation ender x2
    raise KeyboardInterrupt()


conversation_instance = conversation_gen()
converser = lambda x: next(conversation_instance)

if ask_via_input:
    converser = input  ## Alternatively, supply your own inputs

agent_kw = dict(
    llm=llm,
    general_prompt=llama_prompt,
    max_length=4096,
    max_new_tokens=128,
    eos_token_id=[2]
)

agent_ex = AgentExecutor.from_agent_and_tools(
    agent=MyAgent(**agent_kw),
    tools=[AskForInputTool(converser).get_tool(), PythonREPL().get_tool()],
    verbose=False
)

## NOTE: You might want to comment this out to make testing the autograder easier
try:
    agent_ex.run("")
except KeyboardInterrupt:
    print("KeyboardInterrupt")