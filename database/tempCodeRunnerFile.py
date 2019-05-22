@schema
# class Sessions(dj.Manual):
#     definition = """
#     # A session is one behavioural experiment performed on one mouse on one day
#     uid: smallint     # unique number that defines each session
#     session_name: varchar(128)  # unique name that defines each session - YYMMDD_MOUSEID
#     ---
#     -> Mice
#     date: date             # date in the YYYY-MM-DD format
#     experiment_name: varchar(128)  # name of the experiment the session is part of 
#     """

