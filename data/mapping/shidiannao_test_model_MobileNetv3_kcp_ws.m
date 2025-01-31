Network MobileNetV3 {
    Layer conv_stem {
        Type: CONV
        Stride { X: 2, Y: 2 }
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 224, X: 224 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block0_dw {
        Type: CONV
        Stride { X: 2, Y: 2 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 112, X: 112 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block0_pw {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 1, S: 1, Y: 112, X: 112 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block1_pw {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 72, C: 16, R: 1, S: 1, Y: 112, X: 112 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block1_dw {
        Type: CONV
        Stride { X: 2, Y: 2 }
        Dimensions { K: 72, C: 72, R: 3, S: 3, Y: 56, X: 56 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block1_pwl {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 24, C: 72, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block2_pw {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 88, C: 24, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block2_dw {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 88, C: 88, R: 3, S: 3, Y: 28, X: 28 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
    Layer conv_block2_pwl {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 24, C: 88, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
    }
}

// MobileNetV3Features(
//   (conv_stem): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
//   (blocks): Sequential(
//     (0): Sequential(
//       (0): DepthwiseSeparableConv(
//         (conv_dw): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
//         (conv_pw): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
//       )
//     )
//     (1): Sequential(
//       (0): InvertedResidual(
//         (conv_pw): Conv2d(16, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
//         (conv_dw): Conv2d(72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
//         (conv_pwl): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
//       )
//       (1): InvertedResidual(
//         (conv_pw): Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
//         (conv_dw): Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
//         (conv_pwl): Conv2d(88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
//       )
//     )