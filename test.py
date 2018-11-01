import datajoint as dj

dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'simple'


dj.conn()

schema = dj.schema('Database', locals())



@schema
class Mouse(dj.Manual):
      definition = """
      # Mouse definition
      mouse_id: int                  # unique mouse id
      ---
      dob: date                      # mouse date of birth YYYY-MM-DD
      sex: enum('M', 'F', 'U')       # sex of mouse - Male, Female, or Unknown/Unclassified
      """

mouse = Mouse()

# Insert single entry directly with tuple or dict
mouse.insert1( (0, '2017-03-01', 'M') )
data = {
  'mouse_id': 100,
  'dob': '2017-05-12',
  'sex': 'F'
}
mouse.insert1(data)


# Inserting multiple entries
data = [
  (1, '2016-11-19', 'M'),
  (2, '2016-11-20', 'U'),
  (5, '2016-12-25', 'F')
]
mouse.insert(data)
data = [
  {'mouse_id': 10, 'dob': '2017-01-01', 'sex': 'F'},
  {'mouse_id': 11, 'dob': '2017-01-03', 'sex': 'F'},
]

# insert them all
mouse.insert(data)













