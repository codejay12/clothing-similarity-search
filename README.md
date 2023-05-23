<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div id="header" align="center">
<h1>
 

</h1>
  </a>

  <h1 align="center">Clothing Similarity Search</h1>

  <p align="center"
    <a href="https://github.com/othneildrew/Clothing Similarity Search"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Clothing Similarity Search">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Clothing Similarity Search/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Clothing Similarity Search/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#collect-and-preprocess-data"> Collect And Preprocess Data.</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#preprocess-the-text-data">Preprocess the text data</a></li>
      </ul>
    </li>
    <li><a href="#measure-similarity">Measure similarity.</a>
    <ul>
        <li><a href="#extracting-useful-features">Extracting Useful Features</a></li>
        <li><a href="#compute-the-similarity">Compute The Similarity</a></li>
      </ul>
    </li>
    <li><a href="#return-ranked-results">Return Ranked Results</a></li>
    <li><a href="#deploy-the-function">Deploy The Function</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## 🎯About The Project
<p>Project Description:<br>
<p>The objective of this project is to create a machine learning model and implement it as a Google Cloud function to provide a recommendation system for apparel. The solution will take a text string that describes an article of clothing and return a JSON response with a ranked list of links to articles of clothing that are similar on other websites.</p>
<p>Starting with relevant data such as product descriptions, and customer reviews, a machine-learning learning, learning model is trained. The model will be trained to recognize the semantic context of the input text and to identify crucial characteristics of clothing items.</p>
<p>The model will be deployed as a function on the Google Cloud platform when it has been trained. The feature will act as an interface for receiving recommendations-related inquiries. The function will accept a text string as input that describes a piece of clothing, like "blue denim jeans" or "black leather jacket."</p>
<p>The deployed machine learning model will be used by the function to process the input text once it receives a request in order to extract the key features and determine the user's intent. The model will then employ these qualities to browse databases or explore numerous clothes websites in search of comparable things.<p>
<p>A JSON result from the function will offer a ranked list of links to the most pertinent and related clothing items. Based on the model's confidence in each item's resemblance to the input text, the ranking will be determined. The reply is direct link to the store's website.</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Collect and preprocess data -->
##  💾Collect And Preprocess Data

To collect and preprocess data, you likely followed several steps and utilized various techniques. Here is a general description of how you might have accomplished this task:

1. You began by defining the data requirements, determining the specific data needed for your project or analysis. This involved deciding on the data sources, variables of interest, and timeframe for data collection.

2. You identified the data sources from which to obtain the required data. These sources could have included public databases, APIs, web scraping, surveys, or internal company data.

3. You gathered the relevant data from the identified sources. This involved accessing public databases, querying APIs, downloading datasets, or manually collecting data through surveys or other means. You ensured compliance with legal and ethical guidelines, such as obtaining necessary permissions and respecting user privacy.

4. Raw data is often messy and may contain errors, missing values, or inconsistencies. To address this, you performed data cleaning tasks to transform the raw data into a structured format suitable for analysis. These tasks included removing duplicates, handling missing values, correcting errors, standardizing formats, and resolving inconsistencies. You may have utilized software tools, programming languages like Python or R, or data cleaning platforms for this purpose.

5. In some cases, you needed to integrate data from multiple sources to create a comprehensive dataset. This involved merging different datasets based on common variables or using data integration techniques such as joining or concatenating. You paid attention to data integrity and quality during the integration process.

6. Depending on your project requirements, you performed data transformation tasks to convert the data into a different format or structure. This could have involved aggregating data, normalizing or scaling variables, encoding categorical variables, or creating new features through feature engineering techniques.

7. If your dataset contained a large number of variables, you might have performed feature selection to identify the most relevant and informative features for your analysis. You employed feature selection methods such as statistical tests, correlation analysis, or machine learning algorithms to identify the most important variables.

8. In certain cases, you may have needed to sample the data to reduce the dataset's size or balance the class distribution for machine learning tasks. Data sampling techniques like random sampling, stratified sampling, or oversampling/undersampling methods could have been applied to address such requirements.

9. To ensure compatibility with the models or algorithms you planned to use, you formatted the data in a specific way. This involved reshaping the data, converting it into a particular file format (e.g., CSV, JSON, or SQL), or splitting it into training and testing sets.

10. Before proceeding with analysis or modeling, you validated the processed data. This involved checking for any remaining inconsistencies, ensuring the dataset met the desired quality standards, and confirming that it aligned with the initial requirements.

By following these steps, you were able to effectively collect and preprocess data, making it ready for analysis, modeling, or any other downstream tasks in your project. Remember that the specific techniques and tools used can vary depending on the nature of the data and the requirements of your project.
### 🗂️Prerequisites

<p>In my project, I utilized both BeautifulSoup, a powerful web scraping tool, and ParseHub, a comprehensive web scraping software, to extract and parse relevant data from various websites.</p> 

<p>With BeautifulSoup, I employed its intuitive and flexible Python library to scrape web pages and retrieve specific information. By analyzing the HTML structure of the target websites, I navigated through the elements, extracting text, links, images, and other valuable data. BeautifulSoup's rich set of features enabled me to efficiently handle different types of data extraction tasks.</p>

<p>Additionally, I incorporated ParseHub into my project to tackle more complex web scraping requirements. This versatile software allowed me to create custom scraping projects using a visual interface, making it easy to define extraction rules, handle pagination, and handle dynamic web content. ParseHub's advanced features, such as AJAX handling and JavaScript rendering, ensured that I could capture data from websites with dynamic elements or heavy reliance on JavaScript.</p>

<p>By combining the capabilities of BeautifulSoup and ParseHub, I was able to efficiently scrape and parse data from multiple sources, automating the process and saving significant time and effort. The extracted data served as the foundation for my project, enabling me to perform comprehensive analysis, generate insights, and make informed decisions based on the acquired information.</p>

### 📊Preprocess The Text Data

<p>In my project, I employed several preprocessing techniques to clean and prepare the text data for further analysis.</p>

<p>Firstly, I performed data cleaning by removing any irrelevant or redundant information such as HTML tags, special characters, punctuation marks, and numbers. This was achieved by utilizing Python's string manipulation functions and regular expressions, allowing me to systematically eliminate unwanted elements.</p>

<p>Next, I handled text normalization to ensure consistency and standardization across the dataset. This involved converting all text to lowercase to avoid discrepancies caused by capitalization. I also applied techniques such as stemming or lemmatization to reduce words to their base or root form, enabling effective comparison and analysis.</p>

<p>To enhance the quality of the text data, I implemented stop word removal. Stop words, such as "and," "the," or "is," do not carry significant meaning and can be safely excluded from the analysis. By utilizing predefined lists of stop words or creating custom ones, I filtered out these words to focus on the more meaningful content.</p>

<p>Furthermore, I performed tokenization, splitting the text into individual words or tokens. This process facilitated further analysis by treating each word as a separate entity. I used Python's libraries, such as NLTK (Natural Language Toolkit), to tokenize the text efficiently.</p>

<p>Lastly, I addressed the issue of text encoding by ensuring that the data was in a suitable format for analysis. I handled encoding problems, such as converting Unicode characters or non-standard encodings, to ensure compatibility and avoid any misinterpretation of the data.</p>

<p>By employing these preprocessing techniques, I successfully cleaned and standardized the text data in my project, enabling me to perform accurate analysis, text mining, or natural language processing tasks with improved results and insights.</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Measure similarity -->
## 📈Measure similarity

In my project, I employed various techniques to measure similarity between different entities or data points. By utilizing these methods, I was able to quantify the degree of resemblance or likeness between them.


###🤖 Extracting Useful Features
 <p>In my project, I actively extracted useful features by implementing various techniques and methodologies. Firstly, I identified the specific requirements and objectives of my project, determining the key features that were crucial for achieving the desired outcomes.</p>

<p>Using the BeautifulSoup scraping tool and ParseHub software, I employed data extraction techniques to retrieve the relevant information from the web pages. I carefully analyzed the HTML structure and identified the specific elements and attributes that contained the desired features. By leveraging the capabilities of these tools, I extracted text,links, and other data points that were important for my project.</p>

<p>Furthermore, I applied data cleaning and preprocessing techniques to ensure the extracted features were in a usable format. This involved removing any irrelevant or noisy data, handling missing values, and normalizing the extracted information as needed. By performing these preprocessing steps, I ensured the quality and consistency of the features extracted.</p>

<p>In addition to the basic features, I also employed advanced techniques to extract more complex and meaningful features. This included utilizing regular expressions, pattern matching, and natural language processing algorithms to identify and extract specific patterns, entities, or sentiment from the textual data. By incorporating these techniques, I was able to derive deeper insights and uncover hidden relationships within the extracted features.</p>

<p>Throughout the process, I iteratively refined my feature extraction approach, considering feedback, and adapting to the evolving requirements of my project. By actively engaging in feature selection and extraction, I ensured that the features extracted were relevant, informative, and aligned with the goals of my project. These extracted features became the foundation for subsequent analysis, modeling, and decision-making processes in my project.</p>

### ❇️Compute The Similarity
<p>In my project, I computed the similarity between the input text and the texts in my database using a variety of techniques and algorithms. Firstly, I employed natural language processing (NLP) methods to preprocess both the input text and the texts in the database. This preprocessing involved tasks such as tokenization, stop word removal, stemming or lemmatization, and removing punctuation or special characters.</p>

<p>Next, I utilized a term frequency-inverse document frequency (TF-IDF) approach to represent the texts in a numerical format. TF-IDF calculates the importance of each term in a document by considering its frequency within that document and its rarity across the entire corpus. By applying TF-IDF, I transformed the texts into numerical vectors that captured their semantic content.</p>

<p>To measure the similarity between the input text and the texts in the database, I employed cosine similarity. Cosine similarity calculates the cosine of the angle between two vectors, providing a value between 0 and 1, where 1 represents maximum similarity. By comparing the TF-IDF vectors of the input text and the texts in the database, I computed the cosine similarity scores.</p>

<p>Additionally, I experimented with other similarity measures such as Jaccard similarity or Euclidean distance, depending on the specific requirements of my project. These measures allowed me to assess similarity from different perspectives and cater to different use cases.</p>

<p>By employing these techniques, I was able to quantitatively determine the similarity between the input text and the texts in my database. This similarity computation served as a crucial component of my project, enabling me to retrieve relevant and similar texts, categorize documents, or perform tasks such as clustering or recommendation systems based on text similarity.</p>




<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Return Ranked Results -->
## 📉Return Ranked Results

<p>In my project, I developed a machine learning model capable of receiving a text description of a clothing item and generating a ranked list of links to similar items from various websites. This solution was implemented as a function deployed on Google Cloud, enabling it to accept a text string as input and provide JSON responses with ranked suggestions.</p>

<p>To achieve this, I first gathered a large dataset consisting of clothing item descriptions and their corresponding links from different websites. With this dataset, I trained a machine learning model using a suitable algorithm</p>
<h3>Approach-1</h3>
<p> The first model is based on the tfidf ,where it lacks the semantic meaning of text</p>

<h3>Approach-2</h3>
<p>The process of latent semantic indexing involves several steps:</p>

<p>1. Corpus Collection: A collection of documents is gathered, typically in the form of a text corpus. This corpus represents the dataset on which LSI will be applied.</p>

<p>2. Term Frequency-Inverse Document Frequency (TF-IDF) Calculation: TF-IDF is computed for each term (word) in the document collection. TF-IDF quantifies the importance of a term within a specific document and across the entire corpus. It assigns higher weights to terms that appear frequently in a document but less frequently in the entire corpus.</p>

<p>3. Singular Value Decomposition (SVD): SVD is applied to the TF-IDF matrix, which decomposes it into three matrices: U, Σ, and V. The U matrix represents the relationships between terms and concepts, the Σ matrix contains singular values that represent the importance of these concepts, and the V matrix represents the relationships between documents and concepts.</p>

</p>4. Dimensionality Reduction: The dimensionality of the matrices resulting from SVD is reduced by keeping only the top-k singular values and their corresponding columns in U, Σ, and V. This step helps to remove noise and focus on the most significant semantic relationships.</p>

<p>5. Query Processing: When a user submits a query, it is transformed into the semantic space by using the reduced matrices. This process allows the system to identify relevant documents that may not contain the exact query keywords but are semantically related.</p>

<p>Using the trained model, I created a function that takes the input text describing a clothing item. The function then processes the text using natural language processing techniques to extract the relevant features and representations. This allows the model to understand the description and identify the key attributes of the clothing item.</p>


<p>Upon ranking the items, the function generates JSON responses containing the ranked suggestions. These responses are then returned to the user, providing them with a list of recommended similar items from different websites.</p>

<p>By deploying this function on Google Cloud, it becomes readily accessible to users. They can make requests by providing a text description, and the function will promptly generate and deliver the ranked suggestions in the form of JSON responses. This solution allows users to effortlessly explore similar clothing items from various sources, assisting them in finding desired items and facilitating their online shopping experience.</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Deploy The Function -->
# 💡 Build and deploy

Command to build the application. PLease remeber to change the project name and application name
```
gcloud builds submit --tag gcr.io/<ProjectName>/<AppName>  --project=<ProjectName>
```

Command to deploy the application
```
gcloud run deploy --image gcr.io/<ProjectName>/<AppName> --platform managed  --project=<ProjectName> --allow-unauthenticated
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## 📞Contact

<p>Your Name - Reesu Jagan</p> 
<p>Email-Id - reesujagan42@gmail.com</p>

Project Link: [clothing similiarity search](https://github.com/codejay12/clothing-similarity-search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


* [Amazon](https://www.amazon.com/)
* [Flipkart](https://www.flipkart.com/)
* [AJIO](https://www.ajio.com/shop/men)
* [Myntra](https://www.myntra.com/)
* [Nykaa](https://www.nykaafashion.com/)
* [Limeroad](https://www.limeroad.com/)
* [Shoppers Stop](https://www.shoppersstop.com/)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/codejay12/clothing-similarity-search/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/reesu-jagan-115401232/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
