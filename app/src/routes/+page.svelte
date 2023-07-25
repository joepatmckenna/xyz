<script>
	import PostCard from '$lib/components/PostCard.svelte';
	export let data;
</script>

{#each data?.posts as post, index}
	<PostCard {post} {index} />
{/each}
