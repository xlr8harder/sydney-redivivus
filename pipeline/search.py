#!/usr/bin/env python3
import argparse,pathlib,sqlite3,json
p=argparse.ArgumentParser(description='Search Sydney parts with SQLite FTS5 query syntax.')
p.add_argument('query');p.add_argument('--role',choices=['assistant','user','all'],default='assistant');p.add_argument('--limit',type=int,default=20);a=p.parse_args()
con=sqlite3.connect(pathlib.Path(__file__).resolve().parents[1]/'work'/'search.sqlite3');con.row_factory=sqlite3.Row
sql='SELECT conversation_id,message_index,part_type,snippet(parts,5,\'<<\',\'>>\',\'…\',40) AS excerpt,source_url,transcript FROM parts WHERE parts MATCH ?';params=[a.query]
if a.role!='all':sql+=' AND role=?';params.append(a.role)
sql+=' LIMIT ?';params.append(a.limit)
for row in con.execute(sql,params):print(json.dumps(dict(row),ensure_ascii=False))
