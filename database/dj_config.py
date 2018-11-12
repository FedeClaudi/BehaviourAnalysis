import datajoint as dj

def start_connection():
    dj.config['database.user'] = 'root'
    dj.config['database.password'] = 'fede123'
    schema = dj.schema('fede_database_181112', locals())
    dj.conn()

# Change password and username
## dj.set_password()
## dj.set_username()
