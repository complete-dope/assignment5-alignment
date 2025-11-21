
import os 
import sys
import json 

def save_to_jsonl_file_in_streaming_manner(file_path,data, to_continue=False, last = False):
    '''
    continue : this tells whether need to append in the data to the file or write from scratch , if true: then append else: scratch 
    '''
    # assuming a list of values to be coming so starting with a list

    current_bytes =  os.path.getsize(file_path)

    if current_bytes == 0 :
        with open(file_path, 'w') as f: # make sure nothing is already present in this file ( this way we rewrite that file / make it empty) the writes in a file happen only when we close a file !
            pass 
        
        with open(file_path, 'w') as f:
            f.write('[\n')

    else:
        # assuming the bytes are coorect till now 
        pass

    string_data = json.dumps(data)
    
    with open(file_path , 'a') as f:
        if not last:
            f.write(string_data + ',\n')
        else:
            f.write(string_data + ',\n]')

