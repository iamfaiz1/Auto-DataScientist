"""
This file stores the string templates that will act as our lightweight "AI Explainer".
"""

TASK_EXPLANATION = """
I have analyzed your dataset. Because you want to predict the **{target_col}** column, 
and it contains {n_unique} unique values of type '{dtype}', I have determined this is a 
**{task_type}** problem. 
"""

MODEL_SELECTION_EXPLANATION = """
I evaluated several lightweight models on your data. I selected **{best_model}** as the champion 
because it achieved the best {metric_name} score of **{score}**. 
This means it was the most reliable at finding patterns in your specific dataset.
"""