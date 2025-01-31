Network BERT_BASE {
    Layer BERT_QK {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 64, R: 1, S: 1, Y: 128, X: 1 }
		Dataflow {
			SpatialMap(70,70) K;
			TemporalMap(64,64) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(1,1) Y;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1,1) X;
			Cluster(84, P);
			SpatialMap(Sz(R),Sz(R)) R;
			TemporalMap(116,116) Y;
			TemporalMap(31,31) K;
			TemporalMap(17,17) C;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(1,1) X;
		}
    }
}
// [['K', 70], ['C', 64], ['R', 1], ['Y', 1], ['S', 1], ['X', 1], ['P', 84], ['R', 1], ['Y', 116], ['K', 31], ['C', 17], ['S', 1], ['X', 1]]