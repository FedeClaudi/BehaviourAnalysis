import datajoint as dj


def start_connection():
    """
    Starts up the datajoint database and signs in with user and passoword + returns the database name
    """
    dbname = 'fede_database_181112'
    dj.config['database.user'] = 'root'
    dj.config['database.password'] = 'tutorial'
    schema = dj.schema(dbname, locals())
    dj.conn()
    return dbname

"""
# Change password and username
## dj.set_password()
## dj.set_username()
"""
