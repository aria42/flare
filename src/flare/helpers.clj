(ns flare.helpers)


(defn transform-type [fn]
  (cond
    (number? fn) :double
    (instance? clojure.lang.IFn$DD fn) :dd-fn
    (instance? clojure.lang.IFn$DDD fn) :ddd-fn
    (instance? clojure.lang.IFn$ODD fn) :odd-fn
    (instance? clojure.lang.IFn$ODDD fn) :oddd-fn
    :else (throw (ex-info "Don't recognize fn"))))
