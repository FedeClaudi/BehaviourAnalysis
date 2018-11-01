import datajoint as dj


schema = dj.schema('Database', locals())


@schema
class Mouse(dj.Manual):
      definition = """
      # Mouse definition
      mouse_id: varchar(128)                  # unique mouse id
      ---
      strain:   varchar(128)                        # genetic strain
      dob: varchar(128)                        # mouse date of birth YYYY-MM-DD
      sex: enum('M', 'F', 'U')         # sex of mouse - Male, Female, or Unknown/Unclassified
      single_housed: enum('Y', 'N')    # single housed or group caged
      """

@schema
class Session(dj.Manual):
     definition = """
     # experiment session
     session_uid: varchar(128)    # unique session identifier
     ---
      -> Mouse
     session_date: varchar(128)            # session date
     experiment: varchar(128)    # name of the experimenter

     software: enum('M', 'B')         # mantis or old behaviour software
     experimenter: varchar(128)    # name of the experimenter
     """


@schema
class BehaviourRecording(dj.Manual):
     definition = """
     # Individual recordings within a session
     -> Session
     recording_number: int               # recording number within a session
     ---
     metadata_path : varchar(128)       # name of the experimenter
     video_path : varchar(128)       # name of the experimenter

     tracked : enum('Y', 'N')   # asdas
     dlc_data: varchar(128)      # path to .h5 produced as a result of dlc tracking
     """


#
# @schema
# class TrackingData(dj.Manual):
#     definition = """
#         # Results of tracking with DLC
#         -> BehaviourRecording
#         ---
#         Metadata: composite
#
#         """
#























