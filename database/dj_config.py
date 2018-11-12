import datajoint as dj

def start_connection():
    dbname = 'fede_database_181112'
    dj.config['database.user'] = 'root'
    dj.config['database.password'] = 'fede123'
    schema = dj.schema(dbname, locals())
    dj.conn()
    return dbname

# Change password and username
## dj.set_password()
## dj.set_username()
