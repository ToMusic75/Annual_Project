$	-�����@}��F��@�I+O_@!���Y7�@$	�IG^<@@��E��8@��ح?!L�Y�e:I@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$/�$۞@q=
ף�[@A+���@Y�G�z#�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�A`��5�@�G�z�?A����Me�@Y?5^�I��@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�I+O_@�n����?A���x�_@Y㥛� ��?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&���Y7�@1�Zd�?A���Ԁ��@Y/�$V��@*	   ���uA2P
Iterator::Model::Prefetch�����@!�'´}�G@)�����@1�'´}�G@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap+�9��@!�a~qoJ@)y�&1��@1h����F@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator^�I��@!P�"ϴ_!@)^�I��@1P�"ϴ_!@:Preprocessing2F
Iterator::Model����,�@!^�䁎�G@)�ʡE��?1�v"�@?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor{�G�zt?!�ɼ���>){�G�zt?1�ɼ���>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 17.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�Zd�<@�hP`�K@1�Zd�?!q=
ף�[@	!       "	!       *	!       2$	�A`���@D����@���x�_@!���Ԁ��@:	!       B	!       J$	8�A`E�@�K|��@㥛� ��?!/�$V��@R	!       Z$	8�A`E�@�K|��@㥛� ��?!/�$V��@JCPU_ONLY