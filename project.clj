(defproject aria42/flare "0.1.0-SNAPSHOT"
  :description "Dynamic Tensor Graph library in Clojure (think PyTorch, DynNet, etc.)"
  :url "http://github.com/aria42/flare"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :global-vars {*warn-on-reflection* true}
  :jvm-opts ["-mx2000m"]
  :aot :all
  :plugins [[lein-codox "0.10.3"]]
  :codox {:metadata {:doc/format :markdown}
          :source-uri "https://github.com/aria42/flare/tree/master/{filepath}#L{line}"}
  :lein-release {:deploy-via :clojars}
  :scm {:name "git"
        :url "https://github.com/aria42/flare"}
  :profiles {:dev {:global-vars {*warn-on-reflection* true
                                 *unchecked-math* :warn-on-boxed}
                   :dependencies [[criterium "0.4.4"]]}}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [uncomplicate/neanderthal "0.17.0"]
                 [org.clojure/tools.cli "0.3.5"]])
