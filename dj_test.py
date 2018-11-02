import datajoint as dj

dj.config['database.host'] = 'tutorial-db.datajoint.io'
dj.config['database.user'] = 'fede12321'
dj.config['databas.password'] = 'BT1dbFx2Zb'

dj.conn()
schema = dj.schema('fede12321_tutorial', locals())


@schema
class Mouse(dj.Manual):
      definition = """
      mouse_id: int                  # unique mouse id
      ---
      dob: date                      # mouse date of birth
      sex: enum('M', 'F', 'U')    # sex of mouse - Male, Female, or Unknown/Unclassified
      """

m = Mouse()
while True:
     m

a =1

