
def printDict(dict):
  for dict_name in dict.keys():
    if isinstance(dict[dict_name],str):
      print(dict_name+' : '+dict[dict_name])
    elif isinstance(dict_name[dict_name],list):
      dict_value = dict_name[dict_name]
      dict_value_list = [str(i) for i in dict_value]
      dict_value_list2str = ''.join(dict_value_list)
      print(dict_name + ' : ' + dict_value_list2str)
if __name__ == "__main__":
  dict_name = {'a':['a','b','c'],'c':'d','e':'f'}
  printDict(dict_name)