if sys.platform != "darwin":
    try:
        dj.__version__
    except:
        try:
            import datajoint as dj
            from database.dj_config import start_connection 
            dbname, _ = start_connection()    
        except:
            print("Could not connect to database")
        else:
            print("Importing tables")
            from database.TablesDefinitionsV4 import *