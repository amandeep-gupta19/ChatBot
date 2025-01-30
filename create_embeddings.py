import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Directory to save files

project_dir = '/Users/amandeepgupta/Desktop/task1'

os.makedirs(project_dir, exist_ok=True)

# Load course texts from the previous step
course_texts = [
    "Course Title: Learn Scratch Programming, Course Description: Scratch Course is the foundation of coding and is a building block of a coding journey., Price: $30, Total Lessons: 16",
    "Course Title: Learn Cloud Computing Basics-AWS, Course Description: This course covers the basics and the most important services on AWS., Price: $30, Total Lessons: 20",
    "Course Title: Learn Mobile Development, Course Description: Mobile application development is the process of creating software applications that run on a mobile device., Price: $30, Total Lessons: 24",
    "Course Title: Learn Core Java Programming Online, Course Description: Java is a very popular high-level, class-based, object-oriented programming language., Price: $30, Total Lessons: 41",
    "Course Title: Learn Robotics, Course Description: A basic understanding of electronics and robotics opens doors for advancement in many careers., Price: $30, Total Lessons: 25",
    "Course Title: Learn JavaScript, Course Description: JavaScript is the most popular programming language in the world, powering modern web applications., Price: $30, Total Lessons: 18",
    "Course Title: Learn Node JS, Course Description: Node.js developers are in high demand. The language is used for everything from web apps to server-side programming., Price: $30, Total Lessons: 18",
    "Course Title: Learn Cloud Computing Advanced-AWS, Course Description: This course takes you from AWS fundamentals to advanced topics in cloud computing on AWS., Price: $35, Total Lessons: 18",
    "Course Title: Python Programming - Beginner, Course Description: Python is a language with simple syntax and powerful libraries, ideal for beginners., Price: $30, Total Lessons: 16",
    "Course Title: Roblox Programming For Beginners, Course Description: Explore game development with Roblox through this beginner-friendly course., Price: $32, Total Lessons: 15",
    "Course Title: Python Programming - Intermediate, Course Description: Take your Python skills to the next level and start building real-world applications., Price: $35, Total Lessons: 16",
    "Course Title: Python Programming - Advanced, Course Description: This course is designed for those who already know Python basics and want to take it further., Price: $30, Total Lessons: 30",
    "Course Title: Python Programming Group Classes - Beginner, Course Description: Group classes for beginner Python learners with interactive sessions., Price: $35, Total Lessons: 16",
    "Course Title: Advanced Roblox Scripting Workshop, Course Description: Unlock the full potential of your Roblox game development skills in this intermediate-level course., Price: $30, Total Lessons: 14",
    "Course Title: Robotics Adventure Awaits, Course Description: An introductory course for kids to get hands-on with robotics, combining fun and learning., Price: $30, Total Lessons: 16",
    "Course Title: Java Project-Based Course, Course Description: This course is designed for intermediate Java students who want to work on real-world projects., Price: $30, Total Lessons: 7",
    "Course Title: Artificial Intelligence Adventures - Chatbot Like ChatGPT (For Kids), Course Description: Learn to build AI chatbots in this comprehensive 10-day course for kids., Price: $30, Total Lessons: 10",
    "Course Title: Python Playground: Create a Hangman Game, Course Description: Learn Python programming by creating a fun and engaging Hangman game., Price: $30, Total Lessons: 8",
    "Course Title: Scratch Playground: Create a Maze Game!, Course Description: A beginner-level Scratch course where kids create their own maze game., Price: $30, Total Lessons: 8",
    "Course Title: Artificial Intelligence Essentials: Summer Bootcamp, Course Description: An engaging 5-day bootcamp exploring the fundamentals of AI., Price: $30, Total Lessons: 5",
    "Course Title: Time Mastery Camp: AI for Jobs, Business, Careers, Course Description: Learn AI applications for productivity and time management in a professional setting., Price: $30, Total Lessons: 11",
    "Course Title: Build Your Own Theme Park in Roblox, Course Description: Learn Roblox Studio fundamentals while creating a fun and interactive theme park., Price: $30, Total Lessons: 8",
    "Course Title: Java Coding Summer Camp for Young Minds, Course Description: A 5-day adventure into programming with Java for kids., Price: $30, Total Lessons: 5",
    "Course Title: AI Camp for Entrepreneurs: Build Business Success, Course Description: This course teaches AI applications for entrepreneurship and business growth., Price: $30, Total Lessons: 7",
    "Course Title: ChatGPT Boot Camp: Basics & Best Uses, Course Description: A 5-day bootcamp teaching the basics of ChatGPT and its best applications., Price: $30, Total Lessons: 5",
    "Course Title: Create-A-Bot: A Project-Based Robotics Exploration, Course Description: Ignite kids' curiosity with hands-on robotics projects in this 5-day camp., Price: $30, Total Lessons: 5",
    "Course Title: Java & Programming Project-Based Bootcamp, Course Description: An introduction to Java and programming basics with a focus on project-based learning., Price: $30, Total Lessons: 8",
    "Course Title: Chatbot Creators: Design a ChatGPT-like AI, Course Description: A 7-day bootcamp for creating a ChatGPT-like AI chatbot., Price: $30, Total Lessons: 7",
    "Course Title: Web Development from Scratch, Course Description: Learn web development from the ground up in this comprehensive course., Price: $30, Total Lessons: 6",
    "Course Title: Summer Camp: Introduction to Python, Course Description: A 7-day camp where kids explore Python programming while working on creative projects., Price: $30, Total Lessons: 7",
    "Course Title: AI Secrets Revealed: Master Productivity Hacks That Will Blow Your Mind! (For Kids), Course Description: Learn to boost productivity using AI tools., Price: $30, Total Lessons: 11",
    "Course Title: Summer Bootcamp: JavaScript - Real Projects, Real Results, Course Description: Dive into JavaScript with real-world projects in a fun summer bootcamp., Price: $30, Total Lessons: 5",
    "Course Title: AI Disruption: Top Entrepreneurs Harnessing AI for Success (For Kids), Course Description: Understand the role of AI in entrepreneurship and explore real-world applications., Price: $30, Total Lessons: 7",
    "Course Title: The AI Writer's Masterclass: Innovation in Creative Writing! (For Kids), Course Description: Learn how AI can enhance creative writing in this 10-day masterclass., Price: $32, Total Lessons: 10",
    "Course Title: Web Development Pro: Intermediate Level, Course Description: Unlock advanced web development techniques in this intermediate course., Price: $30, Total Lessons: 8",
    "Course Title: Scratch Playground: Create a Scroller Game!, Course Description: An intermediate-level course where kids create a scrolling game in Scratch., Price: $30, Total Lessons: 8",
    "Course Title: AI Pro: Creative Writing Camp for Adults, Course Description: Enhance creative writing using AI in this 10-day course designed for adults., Price: $30, Total Lessons: 10",
    "Course Title: Python Playground: Create Your Own Snake Game, Course Description: A fun and interactive Python course where learners create their own Snake game., Price: $30, Total Lessons: 8",
    "Course Title: Build Your Own Calculator Using Python Bootcamp for Kids, Course Description: Learn Python programming by creating a simple calculator in this hands-on bootcamp., Price: $30, Total Lessons: 8",
    "Course Title: Python Playground: Create a Tic Tac Toe Game, Course Description: Create a Tic Tac Toe game using Python in this beginner-friendly course., Price: $30, Total Lessons: 8",
    "Course Title: Scratch Playground: Create a Flappy Bird Game!, Course Description: A fun Scratch course where kids build their own version of the Flappy Bird game., Price: $30, Total Lessons: 8",
    "Course Title: HTML, CSS, JavaScript: 7-Day Summer Bootcamp, Course Description: Learn HTML, CSS, and JavaScript fundamentals in this engaging 7-day bootcamp., Price: $30, Total Lessons: 7",
    "Course Title: Hands-on Java: Project-Based Learning for Coding Novices, Course Description: A 7-day Java camp focused on building projects and learning through hands-on experience., Price: $30, Total Lessons: 7",
    "Course Title: Python Playground: Create a Memory Game, Course Description: Design and develop a memory game using Python in this interactive course., Price: $30, Total Lessons: 8",
    "Course Title: Summer Bootcamp: 5-Day Scratch Programming for Beginners, Course Description: A beginner-level Scratch programming camp designed to teach kids coding fundamentals., Price: $30, Total Lessons: 5",
    "Course Title: 5-Day Summer Camp: Python for Beginners, Course Description: An introductory camp for kids to explore Python programming in a fun and creative environment., Price: $30, Total Lessons: 5",
    "Course Title: Build a To-Do List App with JavaScript, Course Description: A hands-on course where you learn to create a To-Do List app using JavaScript., Price: $30, Total Lessons: 7",
    "Course Title: Introduction to Robotics for Kids, Course Description: Get hands-on with building robots in this 5-day camp designed for young learners., Price: $30, Total Lessons: 5",
    "Course Title: Game Development with Scratch: Create Your Own Adventure Game, Course Description: Learn how to make your own adventure game with Scratch in this 5-day camp., Price: $30, Total Lessons: 5",
    "Course Title: AI Explorer's Camp: How AI Will Shape Our Future, Course Description: Explore AI concepts and understand how AI is shaping the future., Price: $30, Total Lessons: 5",
    "Course Title: 3D Game Development with Unity for Kids, Course Description: An interactive 5-day camp teaching 3D game development using Unity., Price: $30, Total Lessons: 5",
    "Course Title: Web Development with ReactJS for Beginners, Course Description: Learn how to create dynamic and interactive websites using ReactJS in this beginner's course., Price: $30, Total Lessons: 8",
    "Course Title: Python for Data Science: A Beginner's Bootcamp, Course Description: Start your journey into data science by learning Python and key data science concepts., Price: $30, Total Lessons: 6"
    # Add more texts if available from your extraction step
]

# Create embeddings using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(course_texts).toarray()

# Save the embeddings and vectorizer for later use
np.save(os.path.join(project_dir, 'embeddings.npy'), embeddings)
with open(os.path.join(project_dir, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

# Create a FAISS index and add embeddings
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
faiss.write_index(index, os.path.join(project_dir, 'course_index.faiss'))

print("Embeddings and vectorizer saved successfully.")