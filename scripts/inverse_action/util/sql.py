from sqlalchemy import text

def runQueryFile(engine,file_name):
    file = open(file_name)
    return queryString(engine,text(file.read()))

def queryString(engine,sql):
    results = engine.execute(sql).fetchall()
    data = pd.DataFrame(results)
    if len(data) != 0:
        data.columns = results[0].keys()
    return data
