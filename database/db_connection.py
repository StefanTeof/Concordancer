import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
load_dotenv() 

def connect_to_database():
    try:
        # Assuming you have set environment variables for database connection info
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),        # Environment variable for host
            database=os.getenv('MYSQL_DATABASE'), # Environment variable for database
            user=os.getenv('MYSQL_USER'),        # Environment variable for user
            password=os.getenv('MYSQL_PASSWORD') # Environment variable for password
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print("Successfully connected to MySQL Server version ", db_info)
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
    
        return connection

    except Error as e:
        print("Error while connecting to MySQL", e)
