$	l��� �@��U��@j�t���@!�O��� �@$	80�SW@5� ��u@?H�x�JV@!�'ąj]X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j�t���@?5^�I�?A��ʡ�I@Y�l��))�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�O��� �@+��η?A�A`��l�@Y��ʡ*�@*	   ��sA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator��x�V�@!*?�J�I@)��x�V�@1*?�J�I@:Preprocessing2P
Iterator::Model::Prefetch�ʡE&�@!��x}�H@)�ʡE&�@1��x}�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap%����@!���bsI@)�� �rh�?1��+(�*x?:Preprocessing2F
Iterator::ModelP��n2�@!\m\���H@)R���Q�?1��*��H>?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor�I+��?!0�!�?)�I+��?10�!�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 90.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�/�$�?�rfzP�?+��η?!?5^�I�?	!       "	!       *	!       2$	�$���@!����@��ʡ�I@!�A`��l�@:	!       B	!       J$	�G�:�@��`b�Y�@�l��))�@!��ʡ*�@R	!       Z$	�G�:�@��`b�Y�@�l��))�@!��ʡ*�@JCPU_ONLY