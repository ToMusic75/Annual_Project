$	�K7��\�@YaՔ�@���(\�d@!u�V3�@$	J�
�)W@Td��@[^
!V@!��jg:X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�� �r!�@q=
ף�K@A�/�$L@Y����xŌ@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D�l���@�I+��?A��x�&<@Y�A`��8�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&���(\�d@+�����?A��Q��@Y�O��n�c@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u�V3�@�I+��?Am���q��@Y��C����@*	   `�ptA2P
Iterator::Model::Prefetch-�����@!�(.pY�H@)-�����@1�(.pY�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap7�A`���@!�0sЖI@)/�$��@1��m��rF@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator)\�����@!��$ @))\�����@1��$ @:Preprocessing2F
Iterator::Model�Q����@!iό/i�H@)��|?5^�?12�M�~??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor����Mbp?!`�T�ۑ�>)����Mbp?1`�T�ۑ�>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 89.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	���S�,@���!P�;@�I+��?!q=
ף�K@	!       "	!       *	!       2$	/�$�s@��{.�@��Q��@!m���q��@:	!       B	!       J$	�G�ޤ@jRY@���@�O��n�c@!��C����@R	!       Z$	�G�ޤ@jRY@���@�O��n�c@!��C����@JCPU_ONLY