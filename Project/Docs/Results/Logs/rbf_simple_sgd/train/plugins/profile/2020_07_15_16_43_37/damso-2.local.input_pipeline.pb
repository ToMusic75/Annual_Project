$	��Η��@�m�����@F����hd@!NbX��@$	��-~;MX@� ��@�1̆7W@!���L�X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��/� �@��"���P@AbX9�#@Y`��"ۡ�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F����΋@;�O��n�?A/�$@Y��K7��@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F����hd@`��"���?A���(\�@YV-��d@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&NbX��@
ףp=
�?A�Q���K@Y���M2��@*	   pVAtA2P
Iterator::Model::PrefetchD�l���@!V5`�H@)D�l���@1V5`�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�t�t��@!1���I@)q=
�S�@1F�Q�H�F@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorD�l��!�@!�B*�Z�@)D�l��!�@1�B*�Z�@:Preprocessing2F
Iterator::Model�|?5��@!�M%+�H@)�����M�?1��x��F?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor����Mbp?!5��R|��>)����Mbp?15��R|��>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 98.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	d;�O��0@ň�)�v@@
ףp=
�?!��"���P@	!       "	!       *	!       2$	�Zd;2@s�����8@���(\�@!�Q���K@:	!       B	!       J$	���1��@�ԔU���@V-��d@!���M2��@R	!       Z$	���1��@�ԔU���@V-��d@!���M2��@JCPU_ONLY