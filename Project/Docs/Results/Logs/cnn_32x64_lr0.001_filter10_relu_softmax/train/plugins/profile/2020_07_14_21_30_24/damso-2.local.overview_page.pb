�$	^�I�ڵ@Y./e�^�@;�O���[@!��ʡ�7�@$	�5�sM@��u��.@�ud��E@!!�:qZ<R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��Q���@)\���(�?A;�O���x@Y�C�l缎@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9��v�ߒ@ףp=
��?AL7�A`_t@Y�p=
׈�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;�O���[@�������?AbX9��L@Y�G�z�J@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&��ʡ�7�@�rh��|�?Ao��z��@Y㥛İ��@*	   �S�tA2P
Iterator::Model::Prefetcho�����@!�:_E�*H@)o�����@1�:_E�*H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapT㥛� �@!����I@)���(�U�@1�F-��F@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�(\�BU�@!Fm���@)�(\�BU�@1Fm���@:Preprocessing2F
Iterator::Model�z����@!?s�*H@)B`��"۹?1'�[�a[>?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor����MbP?!o� �p<�>)����MbP?1o� �p<�>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 47.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	����Mb�?k��s��?�rh��|�?!�������?	!       "	!       *	!       2$	4333�@�/���@bX9��L@!o��z��@:	!       B	!       J$	n�����@�w~�jY�@�G�z�J@!㥛İ��@R	!       Z$	n�����@�w~�jY�@�G�z�J@!㥛İ��@JCPU_ONLY2black"�
host�Your program is HIGHLY input-bound because 47.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 