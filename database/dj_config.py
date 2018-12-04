import datajoint as dj


def start_connection():
    """
    Starts up the datajoint database and signs in with user and passoword + returns the database name
    """
    dbname = './data:/var/lib/mysql'
    dj.config['database.host'] = "127.0.0.1" 
    dj.config['database.user'] = 'root'
    dj.config['database.password'] = 'fede'
    # schema = dj.schema(dbname, locals())
    dj.conn()
    return dbname

    """
    # Change password and username
    ## dj.set_password()
    ## dj.set_username()
    """
    # dj.set_password()


if __name__ == "__main__":
    start_connection()
