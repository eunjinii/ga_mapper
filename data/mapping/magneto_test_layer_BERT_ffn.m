Network BERT_BASE {
    Layer BERT_FFN_1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(70, 70) C;
			TemporalMap(162, 162) K;
			TemporalMap(1, 1) Y;
			TemporalMap(1, 1) X;
			Cluster(78, P);
			SpatialMap(237,237) C;
			TemporalMap(1, 1) Y;
			TemporalMap(387, 387) K;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(1, 1) X;
			TemporalMap(Sz(S),Sz(S)) S;
		}
    }
}