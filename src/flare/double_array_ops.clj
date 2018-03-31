(ns flare.double-array-ops
  (:require [flare.core :as flare]
            [flare.helpers :as helpers])
  (:import
   [clojure.lang IFn$DD IFn$ODD IFn$DDD IFn$ODDD]
   [java.util Arrays]))

(def ^:private +tensor-ops+ {})

(def factory
  (reify

    flare/PTensorFactory
    (get-op [this op-key]
      ((get +tensor-ops+ op-key)))
    (-from [this nums]
      (if (number? (first nums))
        (double-array nums)
        (object-array (map #(flare/-from this %) nums))))
    (-zeros [this shape]
      (if (= (count shape) 1)
        (double-array (first shape))
        (object-array (mapv (fn [_] (flare/-zeros this (rest shape)))
                            (range (first shape))))))))

(extend-protocol flare/Tensor
  (class (double-array 0))

  (factory [this] factory)

  (shape [this] (alength ^doubles this))


  (transform
    ([this get-val]
     (case (helpers/transform-type get-val)
       :double (doto (double-array (alength ^doubles this))
                 (Arrays/fill (double get-val)))
       :dd-fn (amap ^doubles this idx _
                    (let [v (aget ^doubles this idx)]
                      (.invokePrim ^IFn$DD get-val v)))
       :odd-fn (let [dims (long-array 1)]
                 (amap ^doubles this idx _
                       (let [v (aget ^doubles this idx)]
                         (aset dims 0 idx)
                         (.invokePrim ^IFn$ODD get-val dims v))))))
    ([this other get-val]
     (case (helpers/transform-type get-val)
       :ddd-fn (amap ^doubles this idx _
                     (let [v (aget ^doubles this idx)
                           o (aget ^doubles other idx )]
                       (.invokePrim ^IFn$DDD get-val v o)))
       :oddd-fn (let [dims (long-array 1)]
                  (amap ^doubles this idx _
                        (let [v (aget ^doubles this idx)
                              o (aget ^doubles other idx )]
                          (aset dims 0 idx)
                          (.invokePrim ^IFn$ODDD get-val dims v o)))))))

  (transform!
    ([this get-val]
     (case (helpers/transform-type get-val)
       :double (Arrays/fill ^doubles this (double get-val))
       :dd-fn (dotimes [idx (alength ^doubles this)]
                (let [val (aget ^doubles this idx)
                      new-val (.invokePrim ^IFn$DD get-val val)]
                  (aset ^doubles this new-val)))
       :odd-fn (let [dims (long-array 1)]
                 (dotimes [idx (alength ^doubles this)]
                   (let [val (aget ^doubles this idx)
                         _  (aset dims 0 idx)
                         new-val (.invokePrim ^IFn$ODD get-val dims val)]
                     (aset ^doubles this new-val)))))
     this)
    ([this other get-val]
     (case (helpers/transform-type get-val)
       :ddd-fn (dotimes [idx (alength ^doubles this)]
                 (let [val (aget ^doubles this idx)
                       other-val (aget ^doubles other idx)
                      new-val (.invokePrim ^IFn$DDD get-val val other-val)]
                  (aset ^doubles this new-val)))
       :oddd-fn (let [dims (long-array 1)]
                  (dotimes [idx (alength ^doubles this)]
                    (let [val (aget ^doubles this idx)
                          other-val (aget ^doubles other idx)
                          _  (aset dims 0 idx)
                          new-val (.invokePrim ^IFn$ODDD get-val dims val other-val)]
                      (aset ^doubles this new-val)))))
     this))

  (add
    ([this other]
     (flare/add this 1.0 other))
    ([this alpha other]
     (flare/transform
      this
      (let [alpha (double alpha)
            other ^doubles other]
        (fn ^double [^longs dim ^double x]
          (+ (aget ^doubles other (aget dim 0)) (* alpha x)))))))
  (add!
    ([this other] (flare/add! this 1.0 other))
    ([this alpha other]
     (flare/transform!
      this
      (let [alpha (double alpha)
            other ^doubles other]
        (fn ^double [^longs dim ^double x]
          (+ (aget ^doubles other (aget dim 0)) (* alpha x)))))))
  (div
    ([this denom]
     (flare/div this 0.0 denom))
    ([this denom-offset denom]
     (flare/div this 0.0 denom 0.0))
    ([this numer-offset denom denom-offset]
     (let [this ^doubles this
           denom ^doubles denom
           numer-offset (double numer-offset)
           denom-offset (double denom-offset)]
       (flare/transform this denom
           (fn ^double [^double x ^double o]
             (/ (+ x numer-offset) (+ o denom-offset)))))))

  (div!
    ([this denom]
     (flare/div! this 0.0 denom))
    ([this denom-offset denom]
     (flare/div! this 0.0 denom 0.0))
    ([this numer-offset denom denom-offset]
     (let [this ^doubles this
           denom ^doubles denom
           numer-offset (double numer-offset)
           denom-offset (double denom-offset)]
       (flare/transform! this denom
           (fn ^double [^double x ^double o]
             (/ (+ x numer-offset) (+ o denom-offset)))))))


  (mult [this other]
    (flare/transform this other
       (fn ^double [^double x ^double o] (* x o))))
  (mult! [this other]
    (flare/transform! this other
     (fn ^double [^double x ^double o] (* x o))))
  (pow [this exp]
    (flare/transform this
      (fn ^double [^double x] (Math/pow x (double exp)))))
  (pow! [this exp]
    (flare/transform this
      (fn ^double [^double x] (Math/pow x (double exp)))))
  (scale [this alpha]
    (flare/transform this
        (fn ^double [^double x] (* x (double alpha)))))
  (scale! [this alpha]
    (flare/transform! this
        (fn ^double [^double x] (* x (double alpha)))))
  (copy! [this other]
    (System/arraycopy ^doubles other 0 ^doubles this 0 (alength ^doubles this))
    this))

