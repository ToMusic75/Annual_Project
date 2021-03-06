�$	�|?5��@% ��Q�@{�G!#�@!��S���@$	�I��,X@��,���?M"[?)�W@!�p��3kX@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails${�G!#�@)\���0Q@A=
ףp�3@Y
ףp��@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&��S���@bX9�ȶ?A�G�zDi@Yj�t����@*	   P�sA2P
Iterator::Model::Prefetch����lt�@!A�;�H@)����lt�@1A�;�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapV-r�@!��ȯI@)#��~
��@1�k	uBD@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorNbX9;�@!<̾#@)NbX9;�@1<̾#@:Preprocessing2F
Iterator::ModelH�z~t�@!So�7P�H@)H�z�G�?1*ON.�E?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor����Mbp?!�� �� �>)����Mbp?1�� �� �>:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 97.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	-���6A@w-���GH@bX9�ȶ?!)\���0Q@	!       "	!       *	!       2$	\���(�[@DԼ"`@=
ףp�3@!�G�zDi@:	!       B	!       J$	K7�A�t�@_�Fu�@
ףp��@!j�t����@R	!       Z$	K7�A�t�@_�Fu�@
ףp��@!j�t����@JCPU_ONLY2black"�
host�Your program is HIGHLY input-bound because 97.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 