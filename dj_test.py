import datajoint as dj


schema = dj.schema('tutorial', locals())


@schema
class Mouse(dj.Manual):
      definition = """
      mouse_id: int                  # unique mouse id
      ---
      dob: date                      # mouse date of birth
      sex: enum('M', 'F', 'U')    # sex of mouse - Male, Female, or Unknown/Unclassified
      """

m = Mouse()
print(m)

a =1

