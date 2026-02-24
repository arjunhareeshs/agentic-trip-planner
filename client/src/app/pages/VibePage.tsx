import { motion } from 'motion/react';
import { useNavigate } from 'react-router';

export default function VibePage() {
    const navigate = useNavigate();

    const destinations = [
        { title: "Maldive", duration: "5D / 4N", price: "$1,299", img: "https://lh3.googleusercontent.com/aida-public/AB6AXuAMK4iuyBiuUzxt2JAgS-9YP8eS4QqrrcssjOpg0KMO13oVuwGpJdyDa9yNef36j-8OKCVB1-yNc858W0az_OjRQ43RU1iifbCUyPCuXHkYe5ka6FFtfKSmDbfcTKJDs5RKRnFzTXtJGPDXY307ukYFbK3DWOyZrCgmu7FC_p_sDDO98I9LK97cKoqtXUmN17CPj3pPXsKQXfXRubJG_fyqoQ_KYin1Qz56itKvu63essZrVhMOL2jOZR3GBPT15NSGD2xQO_9e2O9s" },
        { title: "Amalfi", duration: "7D / 6N", price: "$2,450", img: "https://lh3.googleusercontent.com/aida-public/AB6AXuDO7U03u-ZIpwdTuq77FUTZxIC_hNU8aKTIyHHtH3QJ83Gb6xE6KF-3ybVlf61e2ok-3hjsT3r_e8rGsA6FOME3OqHL_TML7jPyUDtswtqC5I9J9Dr5BTyjxpck0o1IlWedVbiveOqQfFtKiA1MqYhjnuoZQnnbcVw38vJUJZ6EGauboiqel_94c26ilYBVtFpYVNT0p3AYMBfa8ndi8tp2VFQecvigVBIofr8H6wN-6lttW6MXu99KVII61WHlgL7EUyXM95QA6ZVz" },
        { title: "Alpine", duration: "4D / 3N", price: "$899", img: "https://lh3.googleusercontent.com/aida-public/AB6AXuCQLzcKCtRUdjuoPL3BWDJxzNIUdWPkveo56mbtgNgcL3qRbMSJFgzHFV8jO-IT0j0C13g4_8Hzt02jnWc8dE1p2KIR7OQ6-RfncssvZOfvHZBh6ktSktjXVHk6vp7cmtbMWgpzhPoYnwIXOdSKWY3QJVNpL1ufvm---9OYVc4Pq8vsN7uUMtewKBinRieQXaBrpazcIFVuhxSP8zLo8VKmdPsDcu9bSDGrE-SWyGRPcnDiBc8BuuLyiFc7hWxxa3c9bOOFY-ToF0qf" },
        { title: "Santorini", duration: "6D / 5N", price: "$1,850", img: "https://lh3.googleusercontent.com/aida-public/AB6AXuCFO9vFaZ4V-3h4iup881EgrZXCJrhgfzxaqkod9Ji7ysbLYnMZJGA4WtXpH8oVSua0a1MQ0rzgtw3lLIZLSDIzpEiGB4cKfZxnadPYy0hkWj8e_ezbRtLjD-g9DeJ7aYsltnVYZE5MNI7AuVeSIimVtaJ88os4mQhW3rMDy9_MBAKaX1XkRCHcpjpnbuigwA3OuR_GZVASPbg91im0urdFvEsqjln1b2TLE7WzZFyDwI2UxemBjQFaPcCYstdm3CnR1M76qp6r8F_R" },
        { title: "Bali", duration: "8D / 7N", price: "$1,100", img: "https://lh3.googleusercontent.com/aida-public/AB6AXuCYgm74_fHij0Dx03OiFGPmqygKuMAvL35xU09MRp9efnBpZ1DxSZo9cg8Xvzg8bJkUj7lvCuCoBGw3jVF1_4wn15DS4eU2APimdaTU7eQkx7ONQTzn9p5_l8Ufti2MA2WGpcgj3rw2PFNd9CxqehraDFLepxbqTZbIRrM1WDvBYP_rXVr1gaxFTLmS1m3w4c8ov6vGXHW77oQN0dm24j7Y-QFfZnOOaz0lPaPOjo_U1CoNg592Xb0QTLj8gybRoZ0-H5ZV4P-Q482o" },
    ];

    // Double the destinations for a seamless loop
    const marqueeItems = [...destinations, ...destinations, ...destinations];

    return (
        <div className="font-display bg-[#0F172A] text-slate-100 min-h-screen relative overflow-hidden">
            {/* Background Image */}
            <div className="fixed inset-0 z-0">
                <img
                    alt="Sunset over the beach"
                    className="w-full h-full object-cover brightness-75"
                    src="https://lh3.googleusercontent.com/aida-public/AB6AXuAkyikj69W2zynF1N18aezGekYe7aTLSG5Y5XX9Wn4a7a1VF7VW0D-YkvEs50B5EmMvO9L_5qiQFy1RsT4ajzYnARdWDLzfq7ERBe6pgWjx_0fkVL3Xrhm6lfoDkavazp611kEf-TuNV52Nck2FXhKkj9ZZEVIn0yKn3hrZudShUoCnGL9lrX_m5H6hnMazNlryivHuiswHpD02pM0ufgbepQyEKJsvww4LjYcdFzUJ7E35q1GajyFg5WCP8dawmyXmU8ajpRK160vj"
                />
                <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-transparent to-black/80"></div>
            </div>

            <main className="relative z-10 flex flex-col h-screen w-full">
                <div className="flex-grow"></div>

                {/* Marquee Train Section */}
                <div className="flex-none w-full h-[50dvh] flex items-center overflow-hidden">
                    <motion.div
                        className="flex gap-6 px-3"
                        animate={{
                            x: ["0%", "-33.33%"], // Move by one set of the items
                        }}
                        transition={{
                            duration: 20,
                            repeat: Infinity,
                            ease: "linear",
                        }}
                    >
                        {marqueeItems.map((dest, i) => (
                            <div
                                key={`${dest.title}-${i}`}
                                className="w-[300px] h-[400px] flex-shrink-0"
                            >
                                <motion.div
                                    whileHover={{ scale: 1.02, transition: { duration: 0.3 } }}
                                    className="glass h-full rounded-[2.5rem] p-3 flex flex-col transition-all duration-300"
                                >
                                    <div className="h-1/2 w-full overflow-hidden rounded-[2rem]">
                                        <img
                                            alt={dest.title}
                                            className="w-full h-full object-cover"
                                            src={dest.img}
                                        />
                                    </div>
                                    <div className="h-1/2 flex flex-col items-center justify-center py-3 px-1 text-center">
                                        <div className="space-y-1">
                                            <h3 className="text-white font-bold text-[12px] tracking-widest uppercase truncate">
                                                {dest.title}
                                            </h3>
                                            <p className="text-white/60 text-[10px] font-medium leading-tight">
                                                {dest.duration}
                                            </p>
                                        </div>
                                        <div className="mt-4">
                                            <span className="text-primary font-bold text-[12px]">
                                                {dest.price}
                                            </span>
                                        </div>
                                    </div>
                                </motion.div>
                            </div>
                        ))}
                    </motion.div>
                </div>

                {/* Footer Button Section */}
                <div className="flex-grow flex flex-col items-center justify-center pb-10">
                    <motion.button
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        onClick={() => navigate('/result')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="vibe-button bg-primary text-white font-bold py-4 px-8 rounded-full flex items-center space-x-3 transition-transform shadow-lg z-30"
                    >
                        <span className="material-symbols-outlined text-2xl">auto_awesome</span>
                        <span className="uppercase tracking-[0.2em] text-[10px] font-bold">Track your vibe</span>
                    </motion.button>
                </div>
            </main>

            <style>{`
        .glass {
          background: rgba(255, 255, 255, 0.12);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .vibe-button {
          box-shadow: 0 0 40px rgba(249, 115, 22, 0.4);
        }
        .material-symbols-outlined {
          font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }
      `}</style>
        </div>
    );
}
