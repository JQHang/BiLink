from datetime import datetime

def time_values_reformat(time_values_list, src_formats, dst_formats):
   """
   Convert time values from source format to destination format
   """
   reformatted_time_values_list = []
   for time_values in time_values_list:
       dt = datetime.strptime(''.join(time_values), ''.join(src_formats))
       reformatted_time_values = [dt.strftime(fmt) for fmt in dst_formats]
       reformatted_time_values_list.append(reformatted_time_values)
           
   return reformatted_time_values_list