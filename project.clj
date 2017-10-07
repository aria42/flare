(defproject tensors "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :global-vars {*warn-on-reflection* true}
  :aot :all
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.nd4j/nd4j-native "0.9.1"]
                 [org.nd4j/nd4j-api "0.9.1"]
                 [prismatic/schema "1.1.6"]
                 [prismatic/plumbing "0.5.4"]
                 [uncomplicate/neanderthal "0.16.1"]
                 [criterium "0.4.4"]
                 [org.clojure/tools.cli "0.3.5"]])
