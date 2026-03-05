import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, LongType

def add_row_number(sparkSession, df, row_num_col="row_num"):
   """
   Add distributed row numbers to a PySpark DataFrame.
   
   This function uses the classic zipWithIndex approach to assign sequential row numbers
   in a distributed manner.
   
   Parameters
   ----------
   df : pyspark.sql.DataFrame
       Input DataFrame to add row numbers to
   row_num_col : str, default="row_num"
       Name of the column that will contain the row numbers
   
   Returns
   -------
   pyspark.sql.DataFrame
       DataFrame with an additional column containing row numbers (0-indexed)
   """      
   # Apply zipWithIndex to add sequential row numbers
   # This is the most reliable distributed row numbering approach
   rdd_with_index = df.rdd.zipWithIndex()
   
   # Map the RDD to include the index as part of the row
   # x[0] is the original Row, x[1] is the index
   rdd_with_row_num = rdd_with_index.map(lambda x: (*x[0], x[1]))
   
   # Create new schema with the row number column
   new_schema = StructType(
       df.schema.fields + 
       [StructField(row_num_col, LongType(), nullable=False)]
   )
   
   # Convert back to DataFrame with the new schema
   result_df = sparkSession.createDataFrame(rdd_with_row_num, schema=new_schema)
   
   return result_df