dic_error = {'E001':'duplicated tile while parse bbox from def',
             }

dic_warning = {'W001':'test',
             }

dic_info = {'I001': 'test',
               }

def EWI(code_name):
  if code_name in dic_error:
    print('LRZ-Error '+code_name[1:4]+" : "+ dic_error[code_name])
  elif code_name in dic_warning:
    print('LRZ-Warning '+code_name[1:4]+" : "+ dic_warning[code_name])
  elif code_name in dic_info:
    print('LRZ-Info '+code_name[1:4]+" : "+ dic_info[code_name])
  else:
    print('Error: unkown message code')

if __name__ == "__main__":
  EWI('E001')