## We Tried out the following prompts/solution options

########################################################################################
## Task 1

"""
<s>[INST] <<SYS>>
You are a computer science professor who is teaching a class well below his pay grade.
You should respond with valid python code, but can leave documentation as necessary.
<</SYS>>

Please provide two different implementations of fibonacci in python, one that is inefficient and the other
that is efficient. Document your thoughts and explain your reasoning in the comments! [/INST]
"""

########################################################################################
## Task 2

response = generate("""
<s>[INST] <<SYS>>
You are a system that can only respond with valid python code.
You should not generate any discussion output unless it is contained within a comment block.
Do not include any superfluous content, as the responses will go into a REPL editor with no modification.
<</SYS>>

Please provide a basic hello-world application!
[/INST]
```python
## Prints out hello world; defaults to a display in the standard output
print("Hello World!")
```
</s><s>[INST]
Please provide two different implementations of fibonacci in python, one that is inefficient and the other
that is efficient. Document your thoughts and explain your reasoning in block comments!
Add some print statements to show that the implementation works.
[/INST]
```python
""", do_sample=True, eos_token_id=[2, 421, 4954, 7521, 16159, 28956, 29952]).replace('```', '')
