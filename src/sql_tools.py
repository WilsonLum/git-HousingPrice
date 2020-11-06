
# For SQLite library
import sqlite3
from sqlite3 import Error


#-------------------------------------------------------------
# Create database connection
#-------------------------------------------------------------
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return conn


#-------------------------------------------------------------
# List the tables of a database
#-------------------------------------------------------------
def sql_fetch(conn):

    cursorObj = conn.cursor()

    cursorObj.execute('SELECT name from sqlite_master where type= "table"')

    print(cursorObj.fetchall())