try:
    import datajoint as dj
except:
    pass

import sys
if sys.platform == "darwin":
    ip = "192.168.241.87"
else:
    ip = "127.18.0.1" 

def start_connection():
    """
    Starts up the datajoint database and signs in with user and passoword + returns the database name
    
    docker compose yaml file:
    D:\Dropbox (UCL - SWC)\Rotation_vte\mysql-server\docker-compose.yml

    Data are here:
    D:\Dropbox (UCL - SWC)\Rotation_vte\mysql-server\data\Database

    """
    

    dbname = 'DatabaseV4'    # Name of the database subfolder with data
    if dj.config['database.user'] != "root":
        try:
            dj.config['database.host'] = ip
        except:
            print("Could not connect to database")
            return None, None

        dj.config['database.user'] = 'root'
        dj.config['database.password'] = 'fede'
        dj.config['database.safemode'] = False
        dj.config['safemode']= False


        dj.conn()

    schema = dj.schema(dbname)
    return dbname, schema

    """
    # Change password and username
    ## dj.set_password()
    ## dj.set_username()
    """
    # dj.set_password()


def print_erd():
    _, schema = start_connection()
    dj.ERD(schema).draw()


if __name__ == "__main__":
    start_connection()
    # print_erd()
