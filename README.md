# LDA Topic Modeling
Implementation of LDA topic modeling as well as Topic Generation using T-5 for the 2023 SMX Internship.

# Details
This code allows you to generate and visualize a LDA Topic Model, as well as automatically generate titles for topic
groups if so desired using T-5

## Usage
### LDA Model Generation
In order to generate an LDA topic Model:
- In the Data Folder, include any data you wish to process. An example data set containing Military related data is provided as an example.
- In LDATopicModeling.py, update the path of "path_to_json_files" so it points to the data that will be used. Optionally, you can change the names of any output files in LDATopicModelling.py as well.
- Go to the "DataPrep.py" script in "Model Generation" and ensure the code is correct such that data is saved in the format of List[articles: str]
- Ensure that virtualenv "LDAenv" is being used, or create your own virtual environment using the requirements.txt
- Adjust any other parameter in LDATopicModeling.py.
- Run the following command in terminal:
<pre>python LDATopicModeling.py -d -cl -co -m</pre>
This will generate and save the Model to ModelGeneration/Model, as well as create a html visualization using pyLDAvis. If you wish to see this visualization in the browser, you can run the following command:
<pre>python LDATopicModel.py -di</pre>
### LDA Topic Name Generation
Once an LDA model has been created, topic names can automatically be generated using T-5.
To do so, enter the follow in terminal:
<pre>python LDATopicModeling.py -t</pre>
This will generate a name per each topic, and save the results in json format under TopicGeneration/Topic