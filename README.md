# GPT4FRC

A side project to help students get more familiar with collaboration by asking an AI about the FIRST Robotics Competition (FRC), and bringing said information to the bigger discussion.

With any AI application, please be aware that the AI is only as smart as the information we provide it. By default, this repository only contains the official Game Manual provided by FIRST. If you would like to provide additional context to the AI, please specify your documents folder and let the AI do its' work.

As of writing, all data is managed by OpenAI, with a local ChromaDB data store. This will eventually be expanded to allow users to utlize other vectorestores such as DeepLake, and other LLMs such as GPT4All or TextGen.

To use this application, Conda is highly recommended for environment management. Install Conda [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Once you have Conda installed, you can create the necessary environment and perform the following actions to use GPT4FRC
```shell
conda create -n gpt4frc python=3.11.4
conda activate gpt4frc
pip install -r requirements.txt
```

Once you have installed your dependencies, you will need to provide an OpenAI token. You can read the docs and get your token [here](https://platform.openai.com/docs/api-reference)

After receiving your token, you can copy/paste the `.env.template` file and replace the value of `OPENAI_API_KEY` with your token. This is a required step to work with OpenAI.

After that, you should have what you need. Open the `gpt4frc_workbench.ipynb` in your Jupyter Notebook of choice, or import the `gpt4frc` Python script to your own script. Web and Console interactive applications will be developed in the future if this project gains traction.

You will notice after a successful (or vague) answer, you may have a `chroma` folder in your local directory. This is expected if you use ChromaDB as your vectorstore.